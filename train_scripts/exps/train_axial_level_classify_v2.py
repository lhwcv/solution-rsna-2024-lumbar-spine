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
from train_scripts.exps.infer_axial_level_classify import Axial_Level_Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--save_dir', type=str, default='./wkdir/cls/axial_level3_only_1seg/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=2e-4)
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

GRAD_ACC = 4
TGT_BATCH_SIZE = 1
BATCH_SIZE = 1#TGT_BATCH_SIZE // GRAD_ACC
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
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),
    A.VerticalFlip(p=0.5),
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

class Axial_Level_Cls_Model(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=3, pretrained=True):
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
        self.lstm = nn.LSTM(fea_dim,
                                  fea_dim // 2,
                                  bidirectional=True,
                                  batch_first=True, num_layers=2)
        self.cond_fea_layer = nn.Linear(fea_dim, fea_dim)
        self.classifier = nn.Linear(fea_dim, 5)
        self.cond_classifier = nn.Linear(fea_dim, 6)

    def forward_test(self, x, sparse_label):
        #x_flip1  = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x_flip2 = torch.flip(x, dims=[-2, ])  # A.VerticalFlip(p=0.5),
        x0, y0 = self.forward_train(x, sparse_label)
        #x1 = self.forward_train(x_flip1, sparse_label)
        x2, y2 = self.forward_train(x_flip2, sparse_label)
        return (x0 + x2) / 2, (y0 + y2) /2

    def forward_train(self, x, sparse_label):
        # b, z_len, 256, 236
        x = F.pad(x, (0, 0, 0, 0, 1, 1))
        # 使用 unfold 函数进行划窗操作
        x = x.unfold(1, 3, 1)  # bs, z_len, 256, 236, 3
        x = x.permute(0, 1, 4, 2, 3)  # bs, z_len, 3, 256, 236
        bs, z_len, c, h, w = x.shape
        x = x.reshape(bs * z_len, c, h, w)
        x = self.model(x).reshape(bs, z_len, -1)
        x0, _ = self.lstm(x)
        x = x + x0
        level_pred = self.classifier(x) #bs, z_len, 5

        level_prob = level_pred.permute(0, 2, 1).unsqueeze(-1)#bs, 5, z_len, 1
        x = self.cond_fea_layer(x).unsqueeze(1) #bs, 1, z_len, 512

        level_prob = level_prob.softmax(dim=2) # across z_len
        x = (level_prob * x).sum(dim=2) #bs, 5, 512
        x = self.cond_classifier(x).reshape(-1, 5, 2, 3) #bs, 5, 2, 3
        x = x.permute(0, 2, 1, 3) # bs, 2_cond, 5_level, 3
        bs = x.shape[0]
        cond_pred = x.reshape(bs, -1)
        return level_pred, cond_pred

    def forward(self, x, sparse_label):
        if self.training:
            return self.forward_train(x, sparse_label)
        return self.forward_test(x, sparse_label)

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
N_LABELS  = 10

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

        #df_valid = df
        model = Axial_Level_Cls_Model(MODEL_NAME, pretrained=True)
        model.to(device)
        model.load_state_dict(
            torch.load(f'{OUTPUT_DIR}/best_fold_{fold}_ema.pt'),strict=False)

        train_ds = Axial_Level_Dataset(
            data_root,
            df_train,  transform=transforms_train)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=N_WORKERS
        )

        valid_ds = Axial_Level_Dataset(
            data_root,
            df_valid,  transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=N_WORKERS
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

        #loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        best_loss = 1e3
        best_loss_ema = 1e3
        best_metric = 0
        best_metric_ema = 0
        es_step = 0

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(1, EPOCHS):

            logger.write(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['img'].cuda()
                    label = tensor_dict['label'].cuda()
                    # b, z_len, 5
                    sparse_label = tensor_dict['sparse_label'].cuda()
                    cls_label = tensor_dict['cls_label'].cuda()

                    with autocast:
                        pred, cond_pred = model(x, sparse_label)
                        #pred = pred.reshape(-1, 5)
                        #label = label.reshape(-1)
                        #loss = loss_criterion(pred, label)
                        loss = loss_criterion(pred, sparse_label)

                        # cls_loss
                        # cls_loss = 0
                        # for col in range(N_LABELS):
                        #     p = cond_pred[:, col * 3:col * 3 + 3]
                        #     gt = cls_label[:, col]
                        #     loss_part = criterion(p, gt) / N_LABELS
                        #     cls_loss = cls_loss + loss_part
                        bs = x.shape[0]
                        cond_pred = cond_pred.reshape(bs*10, 3)
                        cls_label = cls_label.reshape(-1)
                        cls_loss = criterion(cond_pred, cls_label)
                        loss = loss + cls_loss

                        total_loss += loss.item()
                        if GRAD_ACC > 1:
                            loss = loss / GRAD_ACC

                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        print(cond_pred)
                        print(pred)
                        print(sparse_label)
                        print(cls_label)
                        #continue
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
                        model_ema.update(model)
                        if scheduler is not None:
                            scheduler.step()


            train_loss = total_loss / len(train_dl)
            logger.write(f'train_loss:{train_loss:.6f}')

            total_loss = 0
            y_preds = []
            y_preds_ema = []
            labels = []

            y_cls_preds = []
            y_cls_preds_ema = []
            cls_labels = []

            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, tensor_dict in enumerate(pbar):
                        img = tensor_dict['img'].cuda()
                        label = tensor_dict['label'].cuda()
                        # b, z_len, 5
                        sparse_label = tensor_dict['sparse_label'].cuda()
                        cls_label = tensor_dict['cls_label'].cuda()
                        with autocast:
                            pred, cond_pred = model(img, sparse_label)
                            # pred = pred.reshape(-1, 5)
                            # label = label.reshape(-1)
                            # loss = loss_criterion(pred, label)
                            for col in range(N_LABELS):
                                p = cond_pred[:, col * 3:col * 3 + 3]
                                gt = cls_label[:, col]
                                y_cls_preds.append(p.float().cpu())
                                cls_labels.append(gt.cpu())


                            loss = loss_criterion(pred, sparse_label)

                            bs = x.shape[0]
                            cond_pred = cond_pred.reshape(bs * 10, 3)

                            cls_label = cls_label.reshape(-1)
                            cls_loss = criterion(cond_pred, cls_label)

                            # cls_loss = 0
                            # for col in range(N_LABELS):
                            #     p = cond_pred[:, col * 3:col * 3 + 3]
                            #     gt = cls_label[:, col]
                            #
                            #     y_cls_preds.append(p.float().cpu())
                            #     cls_labels.append(gt.cpu())
                            #
                            #     cls_loss = cls_loss + criterion(p, gt) / N_LABELS

                            loss = loss + cls_loss
                            total_loss += loss.item()

                            pred = pred.reshape(-1, 5)
                            label = label.reshape(-1)

                            y_preds.append(pred.cpu())
                            labels.append(label.cpu())

                            y_ema, cond_pred_ema = model_ema.module(img,sparse_label)
                            y_ema = y_ema.reshape(-1, 5)
                            y_preds_ema.append(y_ema.cpu())

                            for col in range(N_LABELS):
                                p = cond_pred_ema[:, col * 3:col * 3 + 3]
                                y_cls_preds_ema.append(p.float().cpu())

            labels = torch.cat(labels)  # n,
            y_preds = torch.cat(y_preds, dim=0).float().softmax(dim=-1).numpy() # n, 5
            y_preds_ema = torch.cat(y_preds_ema, dim=0).float().softmax(dim=-1).numpy()# n, 5

            mask = np.where(labels!=-100)
            y_preds = y_preds[mask]
            y_preds_ema = y_preds_ema[mask]
            labels = labels[mask]

            #calculate acc by sklean
            y_preds = np.argmax(y_preds, axis=1)
            y_preds_ema = np.argmax(y_preds_ema, axis=1)
            acc = accuracy_score(labels, y_preds)
            acc_ema = accuracy_score(labels, y_preds_ema)
            val_loss = total_loss / len(valid_dl)

            y_cls_preds = torch.cat(y_cls_preds, dim=0)
            y_cls_preds_ema = torch.cat(y_cls_preds_ema, dim=0)
            cls_labels = torch.cat(cls_labels)

            met = criterion2(y_cls_preds, cls_labels)
            met_ema = criterion2(y_cls_preds_ema, cls_labels)

            logger.write(f'val_loss:{val_loss:.6f}, '
                         f'loc_acc:{acc:.6f}, loc_acc_ema:{acc_ema:.6f}'
                         f'val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')

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
        scores.append(best_metric_ema)
        exit(0)

logger.write('metric scores: {}'.format(scores))
logger.write('metric avg scores: {}'.format(np.mean(scores)))

