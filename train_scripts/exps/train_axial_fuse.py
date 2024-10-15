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
        self.out_fea_extractor = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 128)
        )

    def forward(self, x, cond, with_emb=False):
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

        if with_emb:
            embeds = self.out_fea_extractor(x)
            # bs, n_cond, 5_level, 128
            embeds = embeds.reshape(b, -1, levels, 128)
            embeds = embeds.permute(0, 2, 1, 3)
            embeds = embeds.reshape(b, 5, -1) # b, 5, n_cond*128

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        if with_emb:
            return x, embeds
        return x


class Axial_Model_25D(nn.Module):
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
        # self.out_fea_extractor = nn.Sequential(
        #     nn.Linear(fea_dim, fea_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(fea_dim, 128)
        # )

    def forward(self, x, cond):
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        # conds = k // 5

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


class Axial_Model_25D_GRU(nn.Module):
    def __init__(self, base_model='densenet201', pool="avg", pretrained=True):
        super(Axial_Model_25D_GRU, self).__init__()

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

    def forward(self, axial_imgs, cond=None, with_emb=False):
        bs, k, n, h, w = axial_imgs.size()
        axial_imgs = axial_imgs.reshape(bs * k * n, 1, h, w)
        x = self.model(axial_imgs)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 3) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 3)

        y = y.reshape(bs, 2, 5, 3)
        y = y.reshape(bs, -1)

        if with_emb:
            embeds = embeds.reshape(bs, 2, 5, 1024)
            embeds = embeds.permute(0, 2, 1, 3).reshape(bs, 5, -1)
            embeds = self.out_fea_extractor(embeds)  # bs, 5, 512
            return y, embeds

        return y


class HybridModel(nn.Module):
    def __init__(self, backbone_sag='densenet201',
                 backbone_axial='densenet201',
                 pretrained=False):
        super().__init__()
        self.sag_model = Sag_Model_25D(backbone_sag, pretrained=pretrained)
        self.axial_model = Axial_Model_25D_GRU(base_model=backbone_axial,
                                               pretrained=pretrained)
        fdim = 512 + 5 * 128
        self.out_linear = nn.Linear(fdim, 6)

    def forward(self, x, axial_x, cond):
        #return self.forward_train(x, axial_x, cond)
        if self.training:
            return self.forward_dynamic(x, axial_x, cond)
        else:
            return self.forward_test(x, axial_x, cond)

    # def forward_train(self, x, axial_x, cond):
    #     sag_pred, sag_emb = self.sag_model.forward(x, cond, with_emb=True)
    #     bs = axial_x.shape[0]
    #     # bs, 3_cond, 5_level, 3
    #     # sag_pred = sag_pred.reshape(bs, 5, 5, 3)
    #     # sag_pred = sag_pred[:, 1:3, :, :]
    #
    #     axial_pred, axial_emb = self.axial_model.forward(axial_x, with_emb=True)
    #     # bs, 2_cond, 5_level, 3
    #     axial_pred = axial_pred.reshape(bs, 2, 5, 3)
    #
    #     # bs, 5_cond, 5_level, 3
    #     ys = axial_pred#torch.cat((sag_pred, axial_pred), dim=1)
    #
    #     ### fuse ####
    #     # bs, 5_level, fdim
    #     fea = torch.cat((sag_emb, axial_emb), dim=-1)
    #     # bs, 5_level, 5_cond, 3
    #     ys2 = self.out_linear(fea).reshape(bs, 5, 2, 3)
    #     # bs, 5_cond, 5_level, 3
    #     ys2 = ys2.permute(0, 2, 1, 3)
    #     # saggital keep independent
    #     #ys[:, 3: ] = (ys[:, 3: ] + ys2[:, 3: ]) / 2
    #     ys = (ys + ys2)/ 2
    #
    #     ys = ys.reshape(bs, -1)
    #     return ys

    def forward_test(self, x, axial_x, cond):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        #axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),
        axial_x_flip_list = []  # A.VerticalFlip(p=0.5),
        for i in range(len(axial_x)):
            axial_x_flip_list.append(torch.flip(axial_x[i], dims=[-2]))

        ys1 = self.forward_dynamic(x, axial_x, cond)
        ys2 = self.forward_dynamic(x_flip, axial_x_flip_list, cond)
        ys = (ys1 + ys2) / 2
        return ys

    def forward_dynamic(self, x, axial_x_list, cond):

        sag_pred, sag_emb = self.sag_model.forward(x, cond, with_emb=True)
        bs = x.shape[0]

        axial_pred, axial_emb = [], []
        for axial_x in axial_x_list:
            axial_x = axial_x.unsqueeze(0)
            _, n_series, c, h, w = axial_x.shape
            n_series = n_series // 10
            axial_x = axial_x.reshape(n_series, 10, c, h, w)
            p, emb = self.axial_model.forward(axial_x, with_emb=True)
            axial_pred.append(p.mean(dim=0, keepdims=True))
            axial_emb.append(emb.mean(dim=0, keepdims=True))

        axial_pred = torch.cat(axial_pred, dim=0)
        axial_emb = torch.cat(axial_emb, dim=0)

        # bs, 2_cond, 5_level, 3
        axial_pred = axial_pred.reshape(bs, 2, 5, 3)

        # bs, 5_cond, 5_level, 3
        ys = axial_pred  # torch.cat((sag_pred, axial_pred), dim=1)

        ### fuse ####
        # bs, 5_level, fdim
        fea = torch.cat((sag_emb, axial_emb), dim=-1)
        # bs, 5_level, 5_cond, 3
        ys2 = self.out_linear(fea).reshape(bs, 5, 2, 3)
        # bs, 5_cond, 5_level, 3
        ys2 = ys2.permute(0, 2, 1, 3)
        # saggital keep independent
        # ys[:, 3: ] = (ys[:, 3: ] + ys2[:, 3: ]) / 2
        ys = (ys + ys2) / 2

        ys = ys.reshape(bs, -1)
        return ys


def collate_fn(batch):
    img = []
    axial_imgs = []
    label = []
    cond = []

    for item in batch:
        img.append(torch.from_numpy(item['img']))
        axial_imgs.append(torch.from_numpy(item['axial_imgs']))
        label.append(torch.from_numpy(item['label']))
        cond.append(torch.from_numpy(item['cond']))

    return {'img': torch.stack(img, dim=0),
            'axial_imgs': axial_imgs, #torch.stack(axial_imgs, dim=0),#axial_imgs,
            'label': torch.stack(label, dim=0),
            'cond': torch.stack(cond, dim=0),
            }

def mixup_data(x1, x2_list, t, alpha=1.0, use_cuda=True):
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
    #mixed_x2 = lam * x2 + (1 - lam) * x2[index, :, :]
    mixed_x2 = []
    for i in range(len(x2_list)):
        axial1 = lam * x2_list[i]
        axial2 = (1- lam) * x2_list[index[i] ]
        n1 = axial1.shape[0]
        n2 = axial2.shape[0]
        if n1 >= n2:
            axial1 = axial1[:n2]
        else:
            axial2 = axial2[:n1]
        axial = axial1 + axial2
        mixed_x2.append(axial)

    t_a, t_b = t, t[index]
    return mixed_x1, mixed_x2, t_a, t_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_sag', type=str, default='densenet161')
    parser.add_argument('--backbone_axial', type=str, default='densenet161')
    parser.add_argument('--save_dir', type=str, default='./wkdir/exp_sag_axial/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--z_imgs', type=int, default=5)
    return parser.parse_args()


args = get_args()

###
EXP_NUM = 624
NOT_DEBUG = True

OUTPUT_DIR = f'{args.save_dir}/{args.backbone_sag}_{args.backbone_axial}_z_imgs_{args.z_imgs}'

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

easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()

aux_info = get_train_study_aux_info(data_root)


transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),

    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Normalize(mean=0.5, std=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

transforms_val = A.Compose([
    A.Normalize(mean=0.5, std=0.5)
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


transforms_train_axial = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

transforms_val_axial = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def collate_fn(batch):
    img = []
    axial_imgs = []
    label = []
    cond = []

    for item in batch:
        img.append(torch.from_numpy(item['img']))
        axial_imgs.append(torch.from_numpy(item['axial_imgs']))
        label.append(torch.from_numpy(item['label']))
        cond.append(torch.from_numpy(item['cond']))

    return {'img': torch.stack(img, dim=0),
            'axial_imgs': axial_imgs, #torch.stack(axial_imgs, dim=0),#axial_imgs,
            'label': torch.stack(label, dim=0),
            'cond': torch.stack(cond, dim=0),
            }

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

        # df_valid = df_valid[df_valid['study_id'].isin(easy_study_ids)]
        model = HybridModel(
            args.backbone_sag,
            args.backbone_axial,
            pretrained=True)

        # model.load_state_dict(
        #     torch.load(f'./{OUTPUT_DIR}/best_wll_model_fold-{fold}_ema.pt'))

        model.to(device)

        train_ds = RSNA24Dataset_Sag_Axial_Cls(
            data_root,
            aux_info,
            df_train,
            phase='train',
            z_imgs=args.z_imgs,
            axial_z_imgs=args.z_imgs,
            with_sag=True,
            with_axial=True,
            transform=transforms_train,
            axial_transform=transforms_train_axial
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            # pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )
        valid_ds = RSNA24Dataset_Sag_Axial_Cls(
            data_root,
            aux_info,
            df_valid, phase='valid',
            z_imgs=args.z_imgs,
            axial_z_imgs=args.z_imgs,
            with_sag=True,
            with_axial=True,
            transform=transforms_val,
            axial_transform=transforms_val_axial
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=8,
            shuffle=False,
            # pin_memory=True,
            drop_last=False,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
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
                    imgs = tensor_dict['img']
                    imgs = imgs.to(device)
                    axial_imgs = tensor_dict['axial_imgs']
                    for j in range(len(axial_imgs)):
                        axial_imgs[j] = axial_imgs[j].to(device)

                    #axial_imgs = axial_imgs.to(device)
                    t = tensor_dict['label'][:, 15:]
                    t = t.to(device)
                    cond = tensor_dict['cond']
                    cond = cond.to(device)

                    with autocast:
                        loss = 0

                        y = model(imgs, axial_imgs, cond)
                        for col in range(N_LABELS):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
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
                        imgs = tensor_dict['img']
                        imgs = imgs.to(device)
                        axial_imgs = tensor_dict['axial_imgs']
                        #axial_imgs = axial_imgs.to(device)
                        for j in range(len(axial_imgs)):
                            axial_imgs[j] = axial_imgs[j].to(device)

                        t = tensor_dict['label'][:, 15:]
                        t = t.to(device)
                        cond = tensor_dict['cond']
                        cond = cond.to(device)
                        with autocast:
                            loss = 0
                            loss_ema = 0
                            do_mixup = np.random.rand() < 0.2
                            if epoch > 6:
                                do_mixup = False

                            if do_mixup:
                                imgs, axial_imgs, t_a, t_b, lam = mixup_data(imgs, axial_imgs, t)

                            y = model(imgs, axial_imgs, cond)

                            for col in range(N_LABELS):
                                pred = y[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())

                                if do_mixup:
                                    gt_a = t_a[:, col]
                                    gt_b = t_b[:, col]
                                    loss_part = mixup_criterion(criterion, pred, gt_a, gt_b, lam) / N_LABELS
                                else:
                                    loss_part = criterion(pred, gt) / N_LABELS

                                loss = loss + loss_part
                                y_pred = pred.float()

                            total_loss += loss.item()

                            y_ema = model_ema.module(imgs, axial_imgs, cond)
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
            # valid_loss_multi = do_eval(fold_preds, fold_gts)
            # logger.write(loss_message(valid_loss_multi))

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
            #break
        logger.write('best metric: {}\n'.format(best_metric_ema))
        logger.write('###' * 10)
        scores.append(best_metric_ema)
        #exit(0)

logger.write('metric scores: {}'.format(scores))
logger.write('metric avg scores: {}'.format(np.mean(scores)))
