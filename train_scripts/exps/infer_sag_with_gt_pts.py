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

from src.data.keypoint.sag_3d import RSNA24Dataset_Sag_Cls_Use_GT_Point
from src.utils.aux_info import get_train_study_aux_info
from src.utils.comm import create_dir, setup_seed
from src.utils.logger import TxtLogger

from hengck.metric import do_eval, loss_message

import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


#### model
class Sag_Model_25D(nn.Module):
    def __init__(self, model_name='densenet201', in_chans=3,
                 n_classes=4, pretrained=False):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.cond_emb = nn.Embedding(5, 64)
        self.cond_fea_extractor = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, fea_dim),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(0.1)
        self.out_fea_extractor = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 128)
        )
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 3)
        )

    def forward(self, x, cond):
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        levels = k // 5

        x = x.reshape(b * k, d, h, w)
        cond = cond.reshape(-1)
        x = self.model(x)
        cond = self.cond_emb(cond)
        cond = self.cond_fea_extractor(cond)
        x = x * cond
        x = self.drop(x)

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_sag', type=str, default='convnext_small.in12k_ft_in1k_384')
    parser.add_argument('--save_dir', type=str, default='./wkdir/exp/sag_with_gt_pts/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--z_imgs', type=int, default=3)
    parser.add_argument('--saggital_fixed_slices', type=int, default=-1)
    parser.add_argument('--z_use_specific', type=int, default=0)
    parser.add_argument('--xy_use_model_pred', type=int, default=0)
    return parser.parse_args()


args = get_args()
z_use_specific = args.z_use_specific == 1
xy_use_model_pred = args.xy_use_model_pred == 1

###
EXP_NUM = 624
NOT_DEBUG = True

OUTPUT_DIR = f'{args.save_dir}/{args.backbone_sag}_z_imgs_{args.z_imgs}_s{args.saggital_fixed_slices}'
if z_use_specific:
    OUTPUT_DIR = OUTPUT_DIR + '_z_use_specific'
if xy_use_model_pred:
    OUTPUT_DIR = OUTPUT_DIR + '_xy_use_model'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = 8620

IMG_SIZE = [128, 128]
IN_CHANS = 3
N_LABELS = 10  # 15
N_CLASSES = 3 * N_LABELS

AUG_PROB = 0.75

N_FOLDS = 5 if NOT_DEBUG else 2
EPOCHS = 25 if NOT_DEBUG else 2

GRAD_ACC = 2
TGT_BATCH_SIZE = args.bs

BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 4

LR = args.base_lr  # * TGT_BATCH_SIZE / 32
WD = 1e-5
AUG = True

######
create_dir(OUTPUT_DIR)
setup_seed(SEED, deterministic=True)
logger = TxtLogger(OUTPUT_DIR + '/log.txt')

###
data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
df = pd.read_csv(f'{data_root}/train.csv')

df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)

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


skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer

pred_dict = {
    'study_id': [],
    'weight': [],
    'mild': [],
    'modr': [],
    'sevr': []
}

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

        # !! use same with patriot
        folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
        train_idx = folds_t2s_each[folds_t2s_each['fold'] != fold].index
        valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index

        folds_t2s_each_train = folds_t2s_each.loc[train_idx].copy().reset_index(drop=True)
        folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)

        df_train = df[df['study_id'].isin(folds_t2s_each_train['study_id'])].copy().reset_index(drop=True)
        df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)

        model = Sag_Model_25D(args.backbone_sag,
                              in_chans=args.z_imgs,
                              pretrained=True)
        model.load_state_dict(
            torch.load(f'./{OUTPUT_DIR}/best_wll_model_fold-{fold}_ema.pt'))

        model.to(device)

        valid_ds = RSNA24Dataset_Sag_Cls_Use_GT_Point(
            data_root,
            aux_info,
            df_valid, phase='valid', transform=transforms_val,
            z_imgs=args.z_imgs,
            saggital_fixed_slices=args.saggital_fixed_slices,
            z_use_specific=z_use_specific,
            xy_use_model_pred=xy_use_model_pred,
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

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        best_loss = 1.1
        best_loss_ema = 1.1
        best_metric = 1.2
        best_metric_ema = 1.2
        best_score = 1.2
        es_step = 0

        total_loss = 0
        y_preds = []
        y_preds_ema = []
        labels = []

        model.eval()
        fold_preds = []
        fold_gts = []
        fold_study_ids = []
        with tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['img']
                    t = tensor_dict['label']
                    cond = tensor_dict['cond']
                    x = x.to(device)
                    t = t.to(device)
                    cond = cond.to(device)
                    for study_id in tensor_dict['study_ids']:
                        fold_study_ids.append(study_id)

                    with autocast:
                        loss = 0
                        loss_ema = 0

                        y = model(x, cond)

                        for col in range(N_LABELS):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]

                            y_pred = pred.float()
                            y_preds.append(y_pred.cpu())
                            labels.append(gt.cpu())

                            loss = loss + criterion(pred, gt) / N_LABELS
                            y_pred = pred.float()

                        total_loss += loss.item()

                        y_ema = model_ema.module(x, cond)
                        for col in range(N_LABELS):
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

        fold_preds = torch.cat(fold_preds, dim=0)
        fold_preds = nn.Softmax(dim=-1)(fold_preds).cpu().numpy()
        fold_gts = torch.cat(fold_gts, dim=0)
        fold_gts = fold_gts.cpu().numpy()

        y_preds = torch.cat(y_preds, dim=0)
        y_preds_ema = torch.cat(y_preds_ema, dim=0)
        labels = torch.cat(labels)

        met = criterion2(y_preds, labels)
        met_ema = criterion2(y_preds_ema, labels)

        val_loss = total_loss / len(valid_dl)

        print(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')
        valid_loss_multi = do_eval(fold_preds, fold_gts)
        print(loss_message(valid_loss_multi))

        n_samples = fold_preds.shape[0]
        for ni in range(n_samples):
            valid_loss_multi = do_eval(fold_preds[ni:ni + 1], fold_gts[ni:ni + 1])
            study_id = fold_study_ids[ni]
            pred_dict['study_id'].append(int(study_id))
            pred_dict['weight'].append(valid_loss_multi[0])
            pred_dict['mild'].append(valid_loss_multi[1])
            pred_dict['modr'].append(valid_loss_multi[2])
            pred_dict['sevr'].append(valid_loss_multi[3])

        scores.append(met_ema)

print('metric scores: {}'.format(scores))
print('metric avg scores: {}'.format(np.mean(scores)))

# pred_dict_final = {}
# key_prefix = 'xy_use_pred_'
# for k, v in pred_dict.items():
#     if k != 'study_id':
#         k = key_prefix + k
#     pred_dict_final[k] = v
#
# df = pd.DataFrame.from_dict(pred_dict_final)
# df.to_csv(f'./{key_prefix}pred.csv', index=False)
