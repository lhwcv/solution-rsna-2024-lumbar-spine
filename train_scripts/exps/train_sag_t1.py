# -*- coding: utf-8 -*-
import os
import gc
import sys
from dataclasses import dataclass

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

from src.data.classification.sag_axial import RSNA24Dataset_Sag_Axial_Cls
from src.models.classification.model_gem import Sag_Model_25D_GEM2
from src.utils.aux_info import get_train_study_aux_info
from src.utils.comm import create_dir, setup_seed
from src.utils.logger import TxtLogger

from hengck.metric import do_eval, loss_message

import warnings

from train_scripts.data_path import DATA_ROOT

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
from train_scripts.v2.models import build_v2_sag_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=8620)
    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--with_patriot_fold', type=int, default=1)
    parser.add_argument('--backbone_sag', type=str, default='convnext_small.in12k_ft_in1k_384')
    parser.add_argument('--with_cond', type=int, default=0)
    parser.add_argument('--with_gru', type=int, default=0)
    parser.add_argument('--with_level_lstm', type=int, default=0)

    parser.add_argument('--save_dir', type=str, default='./wkdir/v20/sag_t1/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--wd', type=int, default=1e-5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--z_imgs', type=int, default=3)
    parser.add_argument('--aug_prob', type=float, default=0.75)
    parser.add_argument('--grad_acc', type=int, default=2)
    parser.add_argument('--with_mixup', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--crop_size_h', type=int, default=128)
    parser.add_argument('--crop_size_w', type=int, default=128)
    parser.add_argument('--resize_to_size', type=int, default=128)
    parser.add_argument('--ema_decay', type=float, default=0.99)

    parser.add_argument('--saggital_fixed_slices', type=int, default=-1)
    parser.add_argument('--z_use_specific', type=int, default=0)
    parser.add_argument('--xy_use_model_pred', type=int, default=0)
    return parser.parse_args()


args = get_args()
z_use_specific = args.z_use_specific==1
xy_use_model_pred = args.xy_use_model_pred==1

with_gru = args.with_gru == 1
with_patriot_fold = args.with_patriot_fold == 1

OUTPUT_DIR = f'{args.save_dir}/' \
             f'{args.backbone_sag}_z_imgs_' \
             f'{args.z_imgs}_seed_{args.seed}_h' \
             f'{args.crop_size_h}_w{args.crop_size_w}'

if with_gru:
    OUTPUT_DIR = OUTPUT_DIR + '_with_gru'
if args.with_cond == 1:
    OUTPUT_DIR = OUTPUT_DIR + '_legacy'

if os.path.exists(OUTPUT_DIR) and args.only_val !=1:
    print('skip: ', OUTPUT_DIR)
    exit(0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = args.seed

N_LABELS = 10
N_CLASSES = 3 * N_LABELS

AUG_PROB = args.aug_prob
N_FOLDS = args.nfolds
EPOCHS = args.epochs
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
data_root = DATA_ROOT
df = pd.read_csv(f'{data_root}/train.csv')


df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)

#easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()

aux_info = get_train_study_aux_info(data_root)

###
transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),

    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    # A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    # A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

transforms_val = A.Compose([
    # A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=0.5, std=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_params(model, learning_rate, decay=1e-2):
    requires_grad_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            requires_grad_params.append((n, p))
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in requires_grad_params if not any(nd in n for nd in no_decay)],
         'weight_decay': decay, "lr": learning_rate},
        {'params': [p for n, p in requires_grad_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": learning_rate}
    ]
    return optimizer_grouped_parameters


def mixup_data(x1, t, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x1.shape[0]  # bs,seq_len,depth
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :, :]
    t_a, t_b = t, t[index]
    return mixed_x1, t_a, t_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer

scores = []
if args.only_val != 1:
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        logger.write('#' * 30)
        logger.write(f'start fold{fold}')
        logger.write('#' * 30)
        # print(len(trn_idx), len(val_idx))
        # df_train = df.iloc[trn_idx]
        # df_valid = df.iloc[val_idx]

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
        #df_valid = df_valid[df_valid['study_id'].isin(easy_study_ids)]


        model = build_v2_sag_model(
            args.backbone_sag,
            with_cond=args.with_cond==1,
            with_gru=with_gru,
            with_level_lstm=args.with_level_lstm==1,
            z_imgs=args.z_imgs,
            pretrained=True
        )
        # model.load_state_dict(
        #     torch.load(f'./{OUTPUT_DIR}/best_wll_model_fold-{fold}_ema.pt'))

        model.to(device)

        train_ds = RSNA24Dataset_Sag_Axial_Cls(
            data_root,
            aux_info,
            df_train, phase='train', transform=transforms_train,
            z_imgs=args.z_imgs,
            saggital_fixed_slices=args.saggital_fixed_slices,
            z_use_specific=z_use_specific,
            xy_use_model_pred=xy_use_model_pred,
            with_axial=False,
            crop_size_h=args.crop_size_h,
            crop_size_w=args.crop_size_w,
            resize_to_size=args.resize_to_size,
           )
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            # pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS,
        )
        valid_ds = RSNA24Dataset_Sag_Axial_Cls(
            data_root,
            aux_info,
            df_valid, phase='valid', transform=transforms_val,
            z_imgs=args.z_imgs,
            saggital_fixed_slices=args.saggital_fixed_slices,
            z_use_specific=z_use_specific,
            xy_use_model_pred=xy_use_model_pred,
            with_axial=False,
            crop_size_h=args.crop_size_h,
            crop_size_w=args.crop_size_w,
            resize_to_size=args.resize_to_size,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=8,
            shuffle=False,
            # pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS
        )

        if fold == 0:
            print('has:', sum(p.numel() for p in model.parameters()), 'params')

        model_ema = ModelEmaV2(model, decay=0.99)

        model_params = get_params(model, LR, decay=WD)

        optimizer = AdamW(model_params, lr=LR)

        warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
        num_total_steps = (EPOCHS) * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)


        best_loss = 1.1
        best_loss_ema = 1.1
        best_metric = 1.2
        best_metric_ema = 1.2
        best_score = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS + 1):
            logger.write(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['img']
                    t = tensor_dict['label']
                    cond = tensor_dict['cond']
                    x = x.to(device)
                    t = t.to(device)
                    cond = cond.to(device)

                    with autocast:
                        loss = 0
                        # do_mixup = np.random.rand() < 0.2
                        # if epoch > 6:
                        #     do_mixup = False
                        do_mixup = False
                        if do_mixup:
                            x, t_a, t_b, lam = mixup_data(x, t)

                        y = model(x, cond)
                        for col in range(5, 15):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
                            if do_mixup:
                                gt_a = t_a[:, col]
                                gt_b = t_b[:, col]
                                loss_part = mixup_criterion(criterion, pred, gt_a, gt_b, lam) / N_LABELS
                            else:
                                loss_part = criterion(pred, gt) / N_LABELS

                            loss = loss + loss_part

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

            model.eval()
            fold_preds = []
            fold_gts = []
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, tensor_dict in enumerate(pbar):
                        x = tensor_dict['img']
                        t = tensor_dict['label']
                        cond = tensor_dict['cond']
                        x = x.to(device)
                        t = t.to(device)
                        cond = cond.to(device)


                        with autocast:
                            loss = 0
                            loss_ema = 0

                            y = model(x, cond)

                            for col in range(5, 15):
                                pred = y[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()

                            total_loss += loss.item()

                            y_ema = model_ema.module(x, cond)
                            for col in range(5, 15):
                                pred = y_ema[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                loss_ema = loss_ema + criterion(pred, gt) / N_LABELS
                                y_pred_ema = pred.float()
                                y_preds_ema.append(y_pred_ema.cpu())

                            bs, _ = y_ema.shape
                            # bs, 5_cond, 5_level, 3
                            y_ema = y_ema.reshape(bs, -1, 5, 3)
                            fold_preds.append(y_ema)
                            t = t.reshape(bs, -1, 5)
                            fold_gts.append(t)
            #
            # fold_preds = torch.cat(fold_preds, dim=0)
            # fold_preds = nn.Softmax(dim=-1)(fold_preds).cpu().numpy()
            # fold_gts = torch.cat(fold_gts, dim=0)
            # fold_gts = fold_gts.cpu().numpy()

            y_preds = torch.cat(y_preds, dim=0)
            y_preds_ema = torch.cat(y_preds_ema, dim=0)
            labels = torch.cat(labels)

            met = criterion2(y_preds, labels)
            met_ema = criterion2(y_preds_ema, labels)

            val_loss = total_loss / len(valid_dl)
            # if scheduler is not None:
            #     scheduler.step(val_loss)

            logger.write(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')
            #exit(0)
            #valid_loss_multi = do_eval(fold_preds, fold_gts)
            #logger.write(loss_message(valid_loss_multi))

            if val_loss < best_loss or met < best_metric or met_ema < best_metric_ema:
                es_step = 0

                if device != 'cuda:0':
                    model.to('cuda:0')

                if val_loss < best_loss:
                    logger.write(f'epoch:{epoch}, best weighted_logloss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss
                    # fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                    # torch.save(model.state_dict(), fname)

                if met < best_metric:
                    logger.write(f'epoch:{epoch}, best wll_metric updated from {best_metric:.6f} to {met:.6f}')
                    best_metric = met
                    fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                    #torch.save(model.state_dict(), fname)

                if met_ema < best_metric_ema:
                    logger.write(
                        f'epoch:{epoch}, best wll_metric_ema updated from {best_metric_ema:.6f} to {met_ema:.6f}')
                    best_metric_ema = met_ema
                    fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}_ema.pt'
                    torch.save(model_ema.module.state_dict(), fname)

                if device != 'cuda:0':
                    model.to(device)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    logger.write('early stopping')
                    break
        logger.write('best metric: {}\n'.format(best_metric_ema))
        logger.write('###' * 10)
        scores.append(best_metric_ema)

    logger.write('metric scores: {}'.format(scores))
    logger.write('metric avg scores: {}'.format(np.mean(scores)))

def forward_tta(model, x, cond):
    x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
    x0 = model(x, cond)
    x1 = model(x_flip, cond)
    return (x0 + x1) / 2


if True:#args.only_val == 1:
    fname = f'{OUTPUT_DIR}/labels.npy'
    if os.path.exists(fname):
        print('skip')
        exit(0)
    #### val
    cv = 0
    y_preds = []
    y_preds_ema = []
    labels = []
    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion2 = nn.CrossEntropyLoss(weight=weights)
    metric_scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print('#' * 30)
        print(f'start fold{fold}')
        print('#' * 30)

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

        model = build_v2_sag_model(
            args.backbone_sag,
            with_cond=args.with_cond == 1,
            with_gru=with_gru,
            with_level_lstm=args.with_level_lstm == 1,
            z_imgs=args.z_imgs,
            pretrained=True
        )

        model.load_state_dict(
            torch.load(f'./{OUTPUT_DIR}/best_wll_model_fold-{fold}_ema.pt'))

        model.to(device)
        model.eval()
        valid_ds = RSNA24Dataset_Sag_Axial_Cls(
            data_root,
            aux_info,
            df_valid, phase='valid', transform=transforms_val,
            z_imgs=args.z_imgs,
            saggital_fixed_slices=args.saggital_fixed_slices,
            z_use_specific=z_use_specific,
            xy_use_model_pred=xy_use_model_pred,
            with_axial=False,
            crop_size_h=args.crop_size_h,
            crop_size_w=args.crop_size_w,
            resize_to_size=args.resize_to_size,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=8,
            shuffle=False,
            # pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS
        )


        fold_preds = []
        tmp_study_ids = []
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['img'].to(device)
                    t = tensor_dict['label'].to(device)
                    cond = tensor_dict['cond'].to(device)
                    with autocast:
                        ye = forward_tta(model, x, cond)
                        for col in range(5, 15):
                            prede = ye[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
                            y_preds_ema.append(prede.float().cpu())
                            labels.append(gt.cpu())
                        bs, _ = ye.shape
        #break
    y_preds_ema = torch.cat(y_preds_ema,dim=0)
    labels = torch.cat(labels)
    print('y_preds_ema shape: ', y_preds_ema.shape)
    cv = criterion2(y_preds_ema, labels)
    logger.write('cv score ema: {}'.format(cv.item()))

    from sklearn.metrics import log_loss

    y_prede_np = y_preds_ema.softmax(1).numpy()

    labels_np = labels.numpy()
    y_pred_nan = np.zeros((y_preds_ema.shape[0], 1))
    y_prede = np.concatenate([y_pred_nan, y_prede_np], axis=1)

    weights = []
    for l in labels:
        if l == 0:
            weights.append(1)
        elif l == 1:
            weights.append(2)
        elif l == 2:
            weights.append(4)
        else:
            weights.append(0)
    cv2 = log_loss(labels, y_prede, normalize=True, sample_weight=weights)
    print('cv score sklearn: {}'.format(cv2))

    np.save(f'{OUTPUT_DIR}/labels.npy', labels_np)
    np.save(f'{OUTPUT_DIR}/final_pred_ema.npy', y_prede)

