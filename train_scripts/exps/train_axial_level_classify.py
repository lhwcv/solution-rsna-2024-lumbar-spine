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

class Axial_Level_Cls_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 df: pd.DataFrame,
                 transform=None):
        super(Axial_Level_Cls_Dataset, self).__init__()
        self.fns = []
        self.labels = []

        self.img_dir = f'{data_root}/train_images_preprocessed_axial/imgs'
        self.transform = transform
        fns = glob.glob(f'{self.img_dir}/*.npz')
        study_dict_to_fns = {}
        for fn in fns:
            fn = os.path.basename(fn)
            study_id = int(fn.split('_')[0])
            if study_id not in study_dict_to_fns.keys():
                study_dict_to_fns[study_id] = []
            study_dict_to_fns[study_id].append(fn)

        study_ids = df['study_id'].unique().tolist()
        for study_id in tqdm(study_ids):
            if study_id not  in study_dict_to_fns:
                print(f'{study_id} hase no samples')
                continue
            fns = study_dict_to_fns[study_id]
            for fn in fns:
                level = fn.split('level_')[-1].split('_')[0]
                level = int(level)
                self.fns.append(fn)
                self.labels.append(level)
        print('samples: ', len(self.fns))

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        fn = os.path.join(self.img_dir, fn)
        label = self.labels[idx]
        img = np.load(fn)['arr_0']
        if self.transform is not None:
            img = self.transform(image=img)['image']
        img = img.transpose(2, 0, 1).astype(np.float32)
        return {
            'img': img,
            'label': label
        }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='densenet161')
    parser.add_argument('--save_dir', type=str, default='./wkdir/cls/axial_level/')
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
TGT_BATCH_SIZE = 128
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
    A.HorizontalFlip(p=0.5),
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
                 in_chans=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=5,
            global_pool='avg'
        )
    def forward_test(self, x):
        x_flip1  = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x_flip2 = torch.flip(x, dims=[-2, ])  # A.VerticalFlip(p=0.5),
        x0 = self.forward_train(x)
        x1 = self.forward_train(x_flip1)
        x2 = self.forward_train(x_flip2)
        return (x0 + x1 + x2) /3

    def forward_train(self, x):
        x = self.model(x)
        return x
    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_test(x)

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

        #df_valid = df

        model = Axial_Level_Cls_Model(MODEL_NAME, pretrained=True)
        model.to(device)
        # model.load_state_dict(
        #     torch.load(f'./wkdir/keypoint/sag_3d_256/densenet161_lr_0.0006/best_fold_{fold}_ema.pt'))

        train_ds = Axial_Level_Cls_Dataset(
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

        valid_ds = Axial_Level_Cls_Dataset(
            data_root,
            df_valid,  transform=transforms_val)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
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

        loss_criterion = nn.CrossEntropyLoss(reduction='mean')

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
                    x = tensor_dict['img'].cuda()
                    label = tensor_dict['label'].cuda()

                    with autocast:
                        pred = model(x)
                        loss = loss_criterion(pred, label)

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
                        model_ema.update(model)
                        if scheduler is not None:
                            scheduler.step()


            train_loss = total_loss / len(train_dl)
            logger.write(f'train_loss:{train_loss:.6f}')

            total_loss = 0
            y_preds = []
            y_preds_ema = []
            labels = []

            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, tensor_dict in enumerate(pbar):
                        img = tensor_dict['img'].cuda()
                        label = tensor_dict['label'].cuda()

                        with autocast:
                            pred = model(img)
                            loss = loss_criterion(pred, label)
                            total_loss += loss.item()
                            y_preds.append(pred.cpu())
                            labels.append(label.cpu())

                            y_ema = model_ema.module(img)
                            y_preds_ema.append(y_ema.cpu())

            y_preds = torch.cat(y_preds, dim=0).float().softmax(dim=-1).numpy() # n, 5
            y_preds_ema = torch.cat(y_preds_ema, dim=0).float().softmax(dim=-1).numpy()# n, 5
            labels = torch.cat(labels) # n,

            #calculate acc by sklean
            y_preds = np.argmax(y_preds, axis=1)
            y_preds_ema = np.argmax(y_preds_ema, axis=1)
            met = accuracy_score(labels, y_preds)
            met_ema = accuracy_score(labels, y_preds_ema)

            val_loss = total_loss / len(valid_dl)



            logger.write(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')

            if val_loss < best_loss or met > best_metric or met_ema > best_metric_ema:
                es_step = 0
                if val_loss < best_loss:
                    logger.write(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss

                if met > best_metric:
                    logger.write(f'epoch:{epoch}, best metric updated from {best_metric:.6f} to {met:.6f}')
                    best_metric = met
                    fname = f'{OUTPUT_DIR}/best_fold_{fold}.pt'
                    #torch.save(model.state_dict(), fname)

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

