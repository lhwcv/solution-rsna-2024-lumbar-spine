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

from src.data.keypoint.sag_3d import RSNA24Dataset_KeyPoint_Sag_3D
from src.models.keypoint.model_3d_keypoint import RSNA24Model_Keypoint_3D_Sag
from train_scripts.data_path import DATA_ROOT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--save_dir', type=str, default='./wkdir/keypoint/sag_3d_256/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=6e-4)
    return parser.parse_args()


args = get_args()

OUTPUT_DIR = f'{args.save_dir}/{args.backbone}_lr_{args.base_lr}'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 1

AUG_PROB = 0.5

N_FOLDS = 5
TRAIN_FOLDS_LIST = [0, 1, 2, 3, 4]
EPOCHS = 35
MODEL_NAME = args.backbone

GRAD_ACC = 2
TGT_BATCH_SIZE = 16
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

LR = args.base_lr
WD = 1e-5

######
create_dir(OUTPUT_DIR)
setup_seed(SEED, deterministic=True)
logger = TxtLogger(OUTPUT_DIR + '/log.txt')

###
data_root = DATA_ROOT
df = pd.read_csv(f'{data_root}/train.csv')

df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)

# easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()
# df = df[df['study_id'].isin(easy_study_ids)]

aux_info = get_train_study_aux_info(data_root)

###
transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),
    #A.HorizontalFlip(p=0.5),
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


def xyz_pixel_err(pred, gt, mask):
    bs, _ = pred.shape
    pred = pred.reshape(bs, 3, 5, 3)
    gt = gt.reshape(bs, 3, 5, 3)
    mask = mask.reshape(bs, 3, 5, 3)

    pred = pred * mask
    gt = gt * mask

    # xy@512, z@16
    pred[:, :, :, :2] = 512 / 4 * pred[:, :, :, :2]
    gt[:, :, :, :2] = 512 / 4 * gt[:, :, :, :2]

    dis_func = nn.L1Loss(reduction='none')
    # t2 xy
    t2_xy_err = dis_func(pred[:, 0, :, :2],
                         gt[:, 0, :, :2]).sum() / (mask[:, 0, :, :2].sum())
    t2_z_err = dis_func(pred[:, 0, :, 2],
                        gt[:, 0, :, 2]).sum() / (mask[:, 0, :, 2].sum())

    t1_xy_err = dis_func(pred[:, 1:, :, :2],
                         gt[:, 1:, :, :2]).sum() / (mask[:, 1:, :, :2].sum())
    t1_z_err = dis_func(pred[:, 1:, :, 2],
                        gt[:, 1:, :, 2]).sum() / (mask[:, 1:, :, 2].sum())
    return t2_xy_err.item(), t2_z_err.item(), t1_xy_err.item(), t1_z_err.item()


skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)

# if args.only_val != 1:
#     study_id_to_pred_keypoints = {}
#     scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
#     for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
#         if fold not in TRAIN_FOLDS_LIST:
#             continue
#         logger.write('#' * 30)
#         logger.write(f'start fold{fold}')
#         logger.write('#' * 30)
#         #print(len(trn_idx), len(val_idx))
#         # df_train = df.iloc[trn_idx]
#         # df_valid = df.iloc[val_idx]
#
#         folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
#         train_idx = folds_t2s_each[folds_t2s_each['fold'] != fold].index
#         valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index
#
#         folds_t2s_each_train = folds_t2s_each.loc[train_idx].copy().reset_index(drop=True)
#         folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)
#
#         df_train = df[df['study_id'].isin(folds_t2s_each_train['study_id'])].copy().reset_index(drop=True)
#         df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)
#
#         #df_valid = df
#
#         model = RSNA24Model_Keypoint_3D_Sag(MODEL_NAME, pretrained=True)
#         model.to(device)
#         model.load_state_dict(
#             torch.load(f'./wkdir/keypoint/sag_3d_256/densenet161_lr_0.0006/best_fold_{fold}_ema.pt'))
#
#         valid_ds = RSNA24Dataset_KeyPoint_Sag_3D(
#             data_root,
#             aux_info,
#             df_valid, phase='valid', transform=transforms_val)
#         valid_dl = DataLoader(
#             valid_ds,
#             batch_size=BATCH_SIZE * 2,
#             shuffle=False,
#             pin_memory=False,
#             drop_last=False,
#             num_workers=N_WORKERS
#         )
#
#         if fold == 0:
#             print(MODEL_NAME, ' has', sum(p.numel() for p in model.parameters()), 'params')
#
#
#         loss_criterion = nn.SmoothL1Loss(reduction='none')
#         metric_criterion = nn.L1Loss(reduction='none')
#
#         total_loss = 0
#         y_preds = []
#         y_preds_ema = []
#         labels = []
#         masks = []
#         t2_xy_errs, t2_z_errs, t1_xy_errs, t1_z_errs = [], [], [], []
#         model.eval()
#
#
#
#         with tqdm(valid_dl, leave=True) as pbar:
#             with torch.no_grad():
#                 for idx, tensor_dict in enumerate(pbar):
#                     x = tensor_dict['img']
#                     t = tensor_dict['keypoints']
#                     mask = tensor_dict['mask']
#                     study_ids = tensor_dict['study_ids']
#                     x = x.to(device)
#                     t = t.to(device)
#                     mask = mask.to(device)
#
#                     with autocast:
#                         y = model(x)
#                         bs = y.shape[0]
#                         keypoints = y.cpu().numpy().reshape(bs, 3, 5, 3)
#                         for idx, study_id in enumerate(study_ids):
#                             study_id = int(study_id)
#                             if study_id not in study_id_to_pred_keypoints.keys():
#                                 study_id_to_pred_keypoints[study_id] = {}
#                             study_id_to_pred_keypoints[study_id] = {
#                                 'points': keypoints[idx],
#                             }
#
#                         loss = loss_criterion(y * mask, t * mask)
#                         loss = loss.sum() / mask.sum()
#                         total_loss += loss.item()
#
#                         t2_xy_err, t2_z_err, t1_xy_err, t1_z_err = \
#                             xyz_pixel_err(y, t, mask)
#
#                         t2_xy_errs.append(t2_xy_err)
#                         t2_z_errs.append(t2_z_err)
#                         t1_xy_errs.append(t1_xy_err)
#                         t1_z_errs.append(t1_z_err)
#
#                         y_preds.append(y.cpu())
#                         labels.append(t.cpu())
#                         masks.append(mask.cpu())
#
#                         y_ema = model(x)
#                         y_preds_ema.append(y_ema.cpu())
#
#         y_preds = torch.cat(y_preds, dim=0)
#         y_preds_ema = torch.cat(y_preds_ema, dim=0)
#         labels = torch.cat(labels)
#         masks = torch.cat(masks)
#
#         met = metric_criterion(y_preds * masks, labels * masks).sum() / masks.sum()
#         met_ema = metric_criterion(y_preds_ema * masks, labels * masks).sum() / masks.sum()
#
#         val_loss = total_loss / len(valid_dl)
#
#         t2_xy_err = np.array(t2_xy_errs).mean()
#         t2_z_err = np.array(t2_z_errs).mean()
#         t1_xy_err = np.array(t1_xy_errs).mean()
#         t1_z_err = np.array(t1_z_errs).mean()
#
#         print(f't2_xy_err@512:{t2_xy_err:.2f}')
#         print(f't2_z_err@16:{t2_z_err:.2f}')
#         print(f't1_xy_err@512:{t1_xy_err:.2f}')
#         print(f't1_z_err@16:{t1_z_err:.2f}')
#
#         print(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')
#
#     import pickle
#     pickle.dump(study_id_to_pred_keypoints,
#                 open(data_root + f'/v2_sag_3d_keypoints_oof.pkl', 'wb'))
#

models = []
for fold in range(3):
    model = RSNA24Model_Keypoint_3D_Sag(MODEL_NAME, pretrained=True)
    model.to(device)
    model.load_state_dict(
        torch.load(f'./wkdir_final/keypoint_3d_v2_sag/densenet161_lr_0.0006/best_fold_{fold}_ema.pt'))
    model.eval()
    models.append(model)


df_valid = df

valid_ds = RSNA24Dataset_KeyPoint_Sag_3D(
            data_root,
            aux_info,
            df_valid, phase='test', transform=transforms_val)

print('valid_ds len: ', len(valid_ds))
valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=N_WORKERS)

study_id_to_pred_keypoints = {}

with tqdm(valid_dl, leave=True) as pbar:
    with torch.no_grad():
        for idx, tensor_dict in enumerate(pbar):
            x = tensor_dict['img']
            t = tensor_dict['keypoints']
            mask = tensor_dict['mask']
            study_ids = tensor_dict['study_ids']
            x = x.to(device)
            t = t.to(device)
            mask = mask.to(device)

            with autocast:
                keypoints = None
                for i in range(len(models)):
                    y = models[i](x)
                    bs = y.shape[0]
                    if keypoints is None:
                        keypoints = y.cpu().numpy().reshape(bs, 3, 5, 3)
                    else:
                        keypoints += y.cpu().numpy().reshape(bs, 3, 5, 3)
                keypoints = keypoints / len(models)

                for idx, study_id in enumerate(study_ids):
                    study_id = int(study_id)
                    if study_id not in study_id_to_pred_keypoints.keys():
                        study_id_to_pred_keypoints[study_id] = {}
                    study_id_to_pred_keypoints[study_id] = {
                        'points': keypoints[idx],
                    }

import pickle
pickle.dump(study_id_to_pred_keypoints,
                open(data_root + f'/pred_keypoints/v2_sag_3d_keypoints_en3.pkl', 'wb'))