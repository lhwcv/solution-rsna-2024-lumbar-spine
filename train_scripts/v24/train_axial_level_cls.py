# -*- coding: utf-8 -*-
import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import accuracy_score

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
from train_scripts.v24.axial_data import Axial_Level_Dataset_Multi, collate_fn, data_to_cuda
from train_scripts.v24.axial_model import Axial_Level_Cls_Model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--save_dir', type=str, default='./wkdir/v24/pretrain_axial_level_cls/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    return parser.parse_args()


args = get_args()

OUTPUT_DIR = f'{args.save_dir}/{args.backbone}'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 1

AUG_PROB = 0.5

N_FOLDS = 5
TRAIN_FOLDS_LIST = [0, 1]#[0, 1, 2, 3, 4]
EPOCHS = 35
MODEL_NAME = args.backbone

GRAD_ACC = 8
TGT_BATCH_SIZE = 1
BATCH_SIZE = 1  # TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = args.base_lr
WD = 1e-5

######
create_dir(OUTPUT_DIR)
setup_seed(SEED, deterministic=True)
logger = TxtLogger(OUTPUT_DIR + '/log.txt')

###
data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'  # data_root = '/root/autodl-tmp/data/'
df = pd.read_csv(f'{data_root}/train.csv')

df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)

aux_info = get_train_study_aux_info(data_root)

###
transforms_train = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),

    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),
    #
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.3),

    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.CoarseDropout(max_holes=8,
                    max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8, p=AUG_PROB),
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


class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.xy_dis_loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_dict):
        z_label = pred_dict['z_label']
        level_cls_pred = pred_dict['level_cls_pred']
        bs, z_len, _ = level_cls_pred.shape
        z_label = z_label.reshape(-1)
        level_cls_pred = level_cls_pred.reshape(bs * z_len, -1)
        level_cls_loss = self.ce(level_cls_pred, z_label)

        # dense_xy_keypoints = pred_dict['dense_xy_keypoints'] / 128.0
        # xy_pred = pred_dict['xy_pred'] / 128.0
        # dense_xy_mask = pred_dict['dense_xy_mask']
        #
        # dense_xy_keypoints = dense_xy_keypoints * dense_xy_mask
        # xy_pred = xy_pred * dense_xy_mask
        # reg_xy_loss = self.xy_dis_loss_fn(xy_pred, dense_xy_keypoints)

        sparse_pred = pred_dict['sparse_pred']
        sparse_label = pred_dict['sparse_label']
        dense_z_mask = pred_dict['dense_z_mask']
        bce_loss = self.bce(sparse_pred*dense_z_mask,  sparse_label*dense_z_mask)

        total_loss = level_cls_loss + bce_loss# + reg_xy_loss

        # decode z
        xyz_keypoints = pred_dict['xyz_keypoints']
        index_info = pred_dict['index_info']
        coords = pred_dict['coords']

        z_prob = sparse_pred[0].sigmoid()
        z_err = 0
        z_err_n = 0
        # n_seg, 2, 5, 3
        n_seg, _, _, _ = xyz_keypoints.shape

        for n in range(n_seg):
            z0, z1 = index_info[n]
            zs = coords[z0:z1]
            prob = z_prob[z0:z1]
            for i in range(5):
                gt_left_z = xyz_keypoints[n, 0, i, 2]
                gt_right_z = xyz_keypoints[n, 1, i, 2]
                recover_left_z = (prob[:, 0, i] * zs).sum() / prob[:, 0, i].sum()
                recover_right_z = (prob[:, 1, i] * zs).sum() / prob[:, 1, i].sum()
                if gt_left_z > 0:
                    err = torch.abs(gt_left_z - recover_left_z)
                    z_err += err
                    z_err_n += 1
                    # print('err: ', err.item())
                if gt_right_z > 0:
                    err = torch.abs(gt_right_z - recover_right_z)
                    z_err += err
                    z_err_n += 1
                    # print('err: ', err.item())
        z_err = z_err / (z_err_n + 1)

        total_loss += 0.1 * z_err

        return {
            'total_loss': total_loss,
            'level_cls_loss': level_cls_loss,
            'z_err': z_err,
            #'reg_xy_loss': reg_xy_loss
        }


skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)

scores = []
if args.only_val != 1:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        if fold not in TRAIN_FOLDS_LIST:
            continue
        logger.write('#' * 30)
        logger.write(f'start fold{fold}')
        logger.write('#' * 30)

        folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
        train_idx = folds_t2s_each[folds_t2s_each['fold'] != fold].index
        valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index

        folds_t2s_each_train = folds_t2s_each.loc[train_idx].copy().reset_index(drop=True)
        folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)

        df_train = df[df['study_id'].isin(folds_t2s_each_train['study_id'])].copy().reset_index(drop=True)
        df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)

        # df_valid = df
        model = Axial_Level_Cls_Model(MODEL_NAME, pretrained=True)
        model.to(device)

        # try:
        #     model.load_state_dict(
        #         torch.load(f'{OUTPUT_DIR}/best_fold_{fold}_ema.pt'),strict=False)
        # except:
        #     pass

        train_ds = Axial_Level_Dataset_Multi(
            data_root,
            df_train, transform=transforms_train)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )

        valid_ds = Axial_Level_Dataset_Multi(
            data_root,
            df_valid, transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )

        if fold == 0:
            print(MODEL_NAME, ' has', sum(p.numel() for p in model.parameters()), 'params')

        model_ema = ModelEmaV2(model, decay=0.99)

        model_params = get_params(model, LR, decay=WD)
        optimizer = AdamW(model_params, lr=LR)

        warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
        num_total_steps = (EPOCHS + 3) * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)

        # loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        # loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        loss_criterion = LossFunc()

        best_loss = 1e3
        best_loss_ema = 1e3
        best_metric = 0
        best_metric_ema = 0
        es_step = 0

        for epoch in range(1, EPOCHS):

            logger.write(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, tensor_dict in enumerate(pbar):
                    tensor_dict = data_to_cuda(tensor_dict)
                    with autocast:
                        pred_dict = model(tensor_dict)
                        loss_dict = loss_criterion(pred_dict)
                        loss = loss_dict['total_loss']

                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        print(pred_dict['level_cls_pred'])
                        print(pred_dict['z_label'])
                        print(tensor_dict['study_id_list'])
                        # continue
                        sys.exit(1)

                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item()*GRAD_ACC :.5f}',
                            #xy=f"{loss_dict['reg_xy_loss'].item():.4f}",
                            level=f"{loss_dict['level_cls_loss'].item():.4f}",
                            z_err=f"{loss_dict['z_err'].item():.4f}",
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    scaler.scale(loss).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                    if (idx + 1) % GRAD_ACC == 0:

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        model_ema.update(model)
                        if scheduler is not None:
                            scheduler.step()

            train_loss = total_loss / len(train_dl)
            logger.write(f'train_loss:{train_loss:.6f}')

            total_loss = 0
            total_xy_loss = 0
            total_z_loss = 0
            y_preds = []
            y_preds_ema = []
            labels = []

            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, tensor_dict in enumerate(pbar):
                        tensor_dict = data_to_cuda(tensor_dict)
                        with autocast:
                            pred_dict = model_ema.module(tensor_dict)
                            loss_dict = loss_criterion(pred_dict)
                            loss = loss_dict['total_loss']

                            total_loss += loss.item()
                            total_z_loss += loss_dict['z_err'].item()
                            #total_xy_loss += loss_dict['reg_xy_loss'].item()

                            pred = pred_dict['level_cls_pred'].reshape(-1, 5)
                            label = pred_dict['z_label'].reshape(-1)

                            y_preds.append(pred.cpu())
                            labels.append(label.cpu())

            labels = torch.cat(labels)  # n,
            y_preds = torch.cat(y_preds, dim=0).float().softmax(dim=-1).numpy()  # n, 5

            mask = np.where(labels != -100)
            y_preds = y_preds[mask]
            labels = labels[mask]

            # calculate acc by sklean
            y_preds = np.argmax(y_preds, axis=1)
            met_ema = accuracy_score(labels, y_preds)
            val_loss = total_loss / len(valid_dl)
            #xy_loss = total_xy_loss / len(valid_dl)
            z_loss = total_z_loss/ len(valid_dl)

            logger.write(f'val_loss:{val_loss:.6f}, '
                         #f'xy_loss:{xy_loss:.6f}, '
                         f'z_loss:{z_loss:.3f}, '
                         f'loc_acc_ema:{met_ema:.5f}')

            if val_loss < best_loss or met_ema > best_metric_ema:
                es_step = 0
                if val_loss < best_loss:
                    logger.write(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss
                    fname = f'{OUTPUT_DIR}/best_fold_{fold}_best_loss.pt'
                    torch.save(model_ema.module.state_dict(), fname)

                if met_ema > best_metric_ema:
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
        scores.append(best_metric_ema)
        #exit(0)

logger.write('metric scores: {}'.format(scores))
logger.write('metric avg scores: {}'.format(np.mean(scores)))
