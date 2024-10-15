# -*- coding: utf-8 -*-
import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
from glob import glob

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import timm
from transformers import get_cosine_schedule_with_warmup

import albumentations as A
from timm.utils import ModelEmaV2

from sklearn.model_selection import KFold

import argparse

from src.utils.aux_info import get_train_study_aux_info
from src.utils.comm import create_dir, setup_seed
from src.utils.logger import TxtLogger

from train_scripts.v20.sag_data import Sag_2D_Point_Dataset
from train_scripts.v20.models import RSNA24Model_Keypoint_2D


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8620)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--with_patriot_fold', type=int, default=1)

    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--save_dir', type=str, default='./wkdir/v20/sag_t2_2d_keypoints/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=6e-4)
    parser.add_argument('--series_description', type=str, default="T2")
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--wd', type=int, default=1e-5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--grad_acc', type=int, default=2)
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--exclude_hard', type=int, default=1)
    return parser.parse_args()


args = get_args()

with_patriot_fold = args.with_patriot_fold == 1

if args.series_description == 'T2':
    series_description = 'Sagittal T2/STIR'
else:
    series_description = 'Sagittal T1'

OUTPUT_DIR = f'{args.save_dir}/{args.backbone}_lr_{args.base_lr}_sag_{args.series_description}'

exclude_hard = args.exclude_hard == 1
if not exclude_hard:
    OUTPUT_DIR += '_not_exclude_hard'


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = args.seed
AUG_PROB = 0.5

N_FOLDS = args.nfolds
TRAIN_FOLDS_LIST = list(range(N_FOLDS))
EPOCHS = args.epochs
MODEL_NAME = args.backbone

GRAD_ACC = args.grad_acc
TGT_BATCH_SIZE = args.bs
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 4

LR = args.base_lr
WD = args.wd

######
create_dir(OUTPUT_DIR)
setup_seed(SEED, deterministic=True)
logger = TxtLogger(OUTPUT_DIR + '/log.txt')
logger.write(str(args))

###
data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
df = pd.read_csv(f'{data_root}/train.csv')

df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)


###
transforms_train = A.Compose([
    # A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

transforms_val = A.Compose([
    A.Normalize(mean=0.5, std=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_params(model, learning_rate, decay=1e-2):
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': decay, "lr": learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": learning_rate}
    ]
    return optimizer_grouped_parameters


skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)

if args.only_val != 1:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
    scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        if fold not in TRAIN_FOLDS_LIST:
            continue
        logger.write('#' * 30)
        logger.write(f'start fold{fold}')
        logger.write('#' * 30)

        if with_patriot_fold:
            # !! use same with patriot
            folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
            train_idx = folds_t2s_each[folds_t2s_each['fold'] != fold].index
            valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index

            folds_t2s_each_train = folds_t2s_each.loc[train_idx].copy().reset_index(drop=True)
            folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)

            df_train = df[df['study_id'].isin(folds_t2s_each_train['study_id'])].copy().reset_index(drop=True)
            df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)

            # df_valid = df_valid[df_valid['study_id'].isin(easy_study_ids)]
        else:
            df_train = df.iloc[trn_idx]
            df_valid = df.iloc[val_idx]

        model = RSNA24Model_Keypoint_2D(MODEL_NAME,
                                        1,
                                        pretrained=True,
                                        num_classes=10)
        model.to(device)

        train_ds = Sag_2D_Point_Dataset(
            data_root,
            df_train,
            transform=transforms_train,
            phase='train',
            img_size=args.image_size,
            series_description=series_description,
            exclude_hard=exclude_hard,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS
        )

        valid_ds = Sag_2D_Point_Dataset(
            data_root,
            df_valid,
            transform=transforms_val,
            phase='valid',
            img_size=args.image_size,
            series_description=series_description,
            exclude_hard=exclude_hard,
        )

        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS
        )

        if fold == 0:
            print(MODEL_NAME, ' has', sum(p.numel() for p in model.parameters()), 'params')

        model_ema = ModelEmaV2(model, decay=args.ema_decay)

        model_params = get_params(model, LR, decay=WD)
        optimizer = AdamW(model_params, lr=LR)

        warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
        num_total_steps = (EPOCHS + 3) * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)

        #loss_criterion = nn.SmoothL1Loss(reduction='none')
        loss_criterion = nn.L1Loss(reduction='none')
        metric_criterion = nn.L1Loss(reduction='none')

        best_loss = 1e3
        best_loss_ema = 1e3
        best_metric = 1e3
        best_metric_ema = 1e3
        es_step = 0

        for epoch in range(1, EPOCHS):

            logger.write(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['imgs']
                    t = tensor_dict['keypoints']
                    mask = tensor_dict['mask']
                    x = x.to(device)
                    t = t.to(device)
                    mask = mask.to(device)

                    with autocast:
                        y = model(x)
                        weight = 512
                        loss = loss_criterion(weight * y * mask, weight * t * mask)
                        loss = loss.sum() / mask.sum()

                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        sys.exit(1)

                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item() * GRAD_ACC:.6f}',
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % GRAD_ACC == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()

                        model_ema.update(model)

            train_loss = total_loss / len(train_dl)
            logger.write(f'train_loss:{train_loss:.6f}')

            total_loss = 0
            y_preds = []
            y_preds_ema = []
            labels = []
            masks = []

            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, tensor_dict in enumerate(pbar):
                        x = tensor_dict['imgs']
                        t = tensor_dict['keypoints']
                        mask = tensor_dict['mask']
                        x = x.to(device)
                        t = t.to(device)
                        mask = mask.to(device)

                        with autocast:
                            y = model(x)

                            loss = metric_criterion(y * mask, t * mask)
                            loss = loss.sum() / mask.sum()
                            total_loss += loss.item()

                            y_preds.append(y.cpu())
                            labels.append(t.cpu())
                            masks.append(mask.cpu())

                            y_ema = model_ema.module(x)
                            y_preds_ema.append(y_ema.cpu())

            y_preds = torch.cat(y_preds, dim=0)
            y_preds_ema = torch.cat(y_preds_ema, dim=0)
            labels = torch.cat(labels)
            masks = torch.cat(masks)

            met = metric_criterion(y_preds * masks, labels * masks).sum() / masks.sum()
            met_ema = metric_criterion(y_preds_ema * masks, labels * masks).sum() / masks.sum()
            weight = 512
            met = weight * met
            met_ema = weight * met_ema

            val_loss = total_loss / len(valid_dl)

            logger.write(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')

            if val_loss < best_loss or met < best_metric or met_ema < best_metric_ema:
                es_step = 0
                if val_loss < best_loss:
                    logger.write(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss

                if met < best_metric:
                    logger.write(f'epoch:{epoch}, best metric updated from {best_metric:.6f} to {met:.6f}')
                    best_metric = met
                    fname = f'{OUTPUT_DIR}/best_fold_{fold}.pt'
                    #torch.save(model.state_dict(), fname)

                if met_ema < best_metric_ema:
                    logger.write(
                        f'epoch:{epoch}, best metric ema updated from {best_metric_ema:.6f} to {met_ema:.6f}')
                    best_metric_ema = met_ema
                    fname = f'{OUTPUT_DIR}/best_fold_{fold}_ema.pt'
                    torch.save(model_ema.module.state_dict(), fname)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break

        logger.write('best_metric_ema: {}'.format(best_metric_ema))
        scores.append(best_metric_ema)

    logger.write(str(args))
    logger.write('metric scores: {}'.format(scores))
    logger.write('metric avg scores: {}'.format(np.mean(scores)))
