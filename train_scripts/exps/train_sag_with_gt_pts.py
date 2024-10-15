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

class Sag_Model_25D_GRU(nn.Module):
    def __init__(self, base_model='densenet201', pool="avg",
                 in_chans=3,
                 pretrained=True):
        super(Sag_Model_25D_GRU, self).__init__()

        self.model = timm.create_model(base_model,
                                       pretrained=pretrained,
                                       num_classes=0, in_chans=1)
        nc = self.model.num_features

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, 3)

        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_fea_extractor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x, cond=None, with_emb=False):
        #print('x shape: ', x.shape)
        bs, k, n, h, w = x.size()
        x = x.reshape(bs * k * n, 1, h, w)
        x = self.model(x)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 3) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 3)

        y = y.reshape(bs, -1, 5, 3)
        y = y.reshape(bs, -1)

        if with_emb:
            embeds = embeds.reshape(bs, 3, 5, 1024)
            embeds = embeds.permute(0, 2, 1, 3).reshape(bs, 5, -1)
            embeds = self.out_fea_extractor(embeds)  # bs, 5, 512
            return y, embeds

        return y

class Sag_Model_T1_T2(nn.Module):
    def __init__(self, model_name='convnext_small.in12k_ft_in1k_384',
                 pretrained=False,
                 in_chans=3,
                 base_model_cls=Sag_Model_25D):
        super(Sag_Model_T1_T2, self).__init__()
        self.sag_t1_model = base_model_cls(model_name,in_chans=in_chans,
                                           pretrained=pretrained)
        self.sag_t2_model = base_model_cls(model_name,in_chans=in_chans,
                                           pretrained=pretrained)

    def forward(self, x, cond):
        if self.training:
            return self.forward1(x, cond)
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        pred = self.forward1(x, cond)
        pred_flip = self.forward1(x_flip, cond)
        return (pred + pred_flip) / 2

    def forward1(self, x, cond):
        t2 = x[:, :5]
        t1 = x[:, 5: 15]
        cond_t2 = cond[:, :5]
        cond_t1 = cond[:, 5:15]
        pred_t2 = self.sag_t2_model(t2, cond_t2)
        pred_t1 = self.sag_t1_model(t1, cond_t1)
        pred = torch.cat((pred_t2, pred_t1), dim=-1)
        return pred


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_sag', type=str, default='convnext_small.in12k_ft_in1k_384')
    parser.add_argument('--save_dir', type=str, default='./wkdir/exp/sag_with_gt_pts/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--z_imgs', type=int, default=3)
    parser.add_argument('--with_gru', type=int, default=0)
    parser.add_argument('--saggital_fixed_slices', type=int, default=-1)
    parser.add_argument('--z_use_specific', type=int, default=0)
    parser.add_argument('--xy_use_model_pred', type=int, default=0)
    return parser.parse_args()


args = get_args()
z_use_specific = args.z_use_specific==1
xy_use_model_pred = args.xy_use_model_pred==1
with_gru = args.with_gru == 1
###
EXP_NUM = 624
NOT_DEBUG = True

OUTPUT_DIR = f'{args.save_dir}/{args.backbone_sag}_z_imgs_{args.z_imgs}_s{args.saggital_fixed_slices}'
if z_use_specific:
    OUTPUT_DIR = OUTPUT_DIR+'_z_use_specific'
if xy_use_model_pred:
    OUTPUT_DIR = OUTPUT_DIR + '_xy_use_model'
if with_gru:
    OUTPUT_DIR = OUTPUT_DIR + '_with_gru'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = 8620

IMG_SIZE = [128, 128]
IN_CHANS = 3
N_LABELS = 15#15
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

easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()

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

        # !! use same with patriot
        folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
        train_idx = folds_t2s_each[folds_t2s_each['fold'] != fold].index
        valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index

        folds_t2s_each_train = folds_t2s_each.loc[train_idx].copy().reset_index(drop=True)
        folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)

        df_train = df[df['study_id'].isin(folds_t2s_each_train['study_id'])].copy().reset_index(drop=True)
        df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)

        #df_valid = df_valid[df_valid['study_id'].isin(easy_study_ids)]

        base_model_cls = Sag_Model_25D
        if with_gru:
            base_model_cls = Sag_Model_25D_GRU
        model = Sag_Model_T1_T2(args.backbone_sag,
                                in_chans=args.z_imgs,
                                pretrained=True,
                                base_model_cls=base_model_cls)
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
                        do_mixup = np.random.rand() < 0.2
                        if epoch > 6:
                            do_mixup = False

                        if do_mixup:
                            x, t_a, t_b, lam = mixup_data(x, t)

                        y = model(x, cond)
                        for col in range(N_LABELS):
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
            # if scheduler is not None:
            #     scheduler.step(val_loss)

            logger.write(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')
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
                    torch.save(model.state_dict(), fname)

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



