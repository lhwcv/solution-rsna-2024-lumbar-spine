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

from src.data.classification.v2_all_view import RSNA24Dataset_Cls_V2#, RSNA24Dataset_Cls_V2_Level_by_Level
from src.data.classification.v2_all_view_cp import RSNA24Dataset_Cls_V2_Multi_Axial
from src.utils.aux_info import get_train_study_aux_info
from src.utils.comm import create_dir, setup_seed
from src.utils.logger import TxtLogger
import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class Model_iafoss(nn.Module):
    def __init__(self, base_model='densenet201', pool="avg", pretrain=True):
        super(Model_iafoss, self).__init__()
        self.base_model = base_model
        NUM_CLS = 15
        self.model = timm.create_model(self.base_model, pretrained=pretrain, num_classes=0, in_chans=3)
        nc = self.model.num_features
        # self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(nc, 512),
        #                           nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLS))

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, NUM_CLS)
        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]

        input1 = input1.reshape(-1, shape[2], shape[3], shape[4])
        x = self.model(input1)

        embeds, _ = self.gru(x.reshape(batch_size, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(batch_size, 5, -1) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(batch_size, 5, -1)

        # """
        embeds = embeds.reshape(batch_size, 1, 1024)
        return y, embeds


#### model
class RSNA24Model(nn.Module):
    def __init__(self, model_name='densenet201', in_c=30,
                 n_classes=4, pretrained=False):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=5,
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
        fea = self.out_fea_extractor(x)

        # bs, 5_cond, 5_level, 128
        fea = fea.reshape(b, 5, levels, 128)

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x, fea


class HybridModel(nn.Module):
    def __init__(self, backbone_sag='densenet201',
                 backbone_axial='densenet201',
                 pretrained=False):
        super().__init__()
        self.sag_model = RSNA24Model(backbone_sag, pretrained=pretrained)
        self.axial_model = Model_iafoss(base_model=backbone_axial, pretrain=pretrained)
        fdim = 1024 + 5 * 128
        self.out_linear = nn.Linear(fdim, 15)

    def forward(self, x, axial_x, cond):
        if self.training:
            return self.forward_train(x, axial_x, cond)
        else:
            return self.forward_test(x, axial_x, cond)

    def forward_train(self, x, axial_x, cond):
        """

        :param x: b, 6, 3, h, w
        :param axial_x: b, d, 3, h, w
        :param cond: b, 5
        :return:
        """

        y, axial_embs = self.axial_model(axial_x)
        ys2, sag_embs = self.sag_model(x, cond)
        b = axial_x.shape[0]
        ys2 = ys2.reshape(b, 5, 1, 3)

        # sag_embs: bs, 5_cond, 1_level, 128
        sag_embs = sag_embs.permute(0, 2, 1, 3).reshape(b, 1, -1)  # bs, 1, 5*128
        embs = torch.cat((sag_embs, axial_embs), dim=-1)
        ys = self.out_linear(embs).reshape(b, 1, 5, 3)  # bs, 1_level, 5_cond, 3
        ys = ys.permute(0, 2, 1, 3)  # # bs, 5_cond, 1_level, 3
        ys = ys + ys2
        ys = ys.reshape(b, -1)
        return ys

    def forward_test(self, x, axial_x, cond):

        with torch.no_grad():
            # b, 5, 6, 3, h, w
            b, k, d, c, h, w = axial_x.shape
            axial_embs = []
            # axial_ys = []
            for ik in range(k):
                input1 = axial_x[:, ik]
                y, emb = self.axial_model(input1)
                # emb shape: bs, 1, 1024
                axial_embs.append(emb)
                # y shape: bs, 5, 3
                # y = y.unsqueeze(2)
                # axial_ys.append(y)

            # bs, 5_level, 1024
            axial_embs = torch.cat(axial_embs, dim=1)
            # bs, 5_cond, 5_level, 3
            # axial_ys = torch.cat(axial_ys, dim=2)  # bs, 5, 5, 3

        ys2, sag_embs = self.sag_model(x, cond)
        ys2 = ys2.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        # ys2[:, 3:, :, :] = axial_ys[:, 3:, :, :]

        # sag_embs: bs, 5_cond, 5_level, 128
        sag_embs = sag_embs.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128
        embs = torch.cat((sag_embs, axial_embs), dim=-1)
        ys = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        ys = ys.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        ys = ys + ys2
        ys = ys.reshape(b, -1)
        return ys

    def forward_axial(self, axial_x, axial_x_flip):
        # b, 5, 6, 3, h, w
        b, k, d, c, h, w = axial_x.shape
        axial_embs = []
        axial_embs_flip = []
        for ik in range(k):
            input1 = axial_x[:, ik]
            _, emb = self.axial_model(input1)
            axial_embs.append(emb)

            input1 = axial_x_flip[:, ik]
            _, emb_flip = self.axial_model(input1)
            axial_embs_flip.append(emb_flip)

        # bs, 5_level, 1024
        axial_embs = torch.cat(axial_embs, dim=1)
        axial_embs_flip = torch.cat(axial_embs_flip, dim=1)
        return  axial_embs, axial_embs_flip

    def forward_test_tta_axial_dynamic(self, x, axial_x_list, cond):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        axial_x_flip_list = []  # A.VerticalFlip(p=0.5),
        for i in range(len(axial_x_list)):
            axial_x_flip_list.append( torch.flip(axial_x_list[i], dims=[-2]) )

        all_axial_embs = []
        all_axial_embs_flip = []
        for i in range(len(axial_x_list)):
            axial_x = axial_x_list[i].unsqueeze(0)
            axial_x_flip = axial_x_flip_list[i].unsqueeze(0)
            axial_embs, axial_embs_flip = self.forward_axial(axial_x, axial_x_flip)
            all_axial_embs.append(axial_embs)
            all_axial_embs_flip.append(axial_embs_flip)

        axial_embs = torch.cat(all_axial_embs, dim=0)
        axial_embs_flip = torch.cat(all_axial_embs_flip, dim=0)

        b = x.shape[0]

        ys2, sag_embs = self.sag_model(x, cond)
        ys2 = ys2.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        ys2_flip, sag_embs_flip = self.sag_model(x_flip, cond)
        ys2_flip = ys2_flip.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        # sag_embs: bs, 5_cond, 5_level, 128
        sag_embs = sag_embs.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128
        sag_embs_flip = sag_embs_flip.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128

        #
        embs = torch.cat((sag_embs, axial_embs), dim=-1)
        out1 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        out1 = out1.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        out1 = out1 + ys2

        embs = torch.cat((sag_embs_flip, axial_embs_flip), dim=-1)
        out4 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        out4 = out4.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        out4 = out4 + ys2_flip

        ys = (out1 + out4) / 2.0
        ys = ys.reshape(b, -1)
        return ys

    def forward_test_tta(self, x, axial_x, cond):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),

        # b, 5, 6, 3, h, w
        b, k, d, c, h, w = axial_x.shape
        axial_embs = []
        axial_embs_flip = []
        for ik in range(k):
            input1 = axial_x[:, ik]
            _, emb = self.axial_model(input1)
            axial_embs.append(emb)

            input1 = axial_x_flip[:, ik]
            _, emb_flip = self.axial_model(input1)
            axial_embs_flip.append(emb_flip)

        # bs, 5_level, 1024
        axial_embs = torch.cat(axial_embs, dim=1)
        axial_embs_flip = torch.cat(axial_embs_flip, dim=1)

        ys2, sag_embs = self.sag_model(x, cond)
        ys2 = ys2.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        ys2_flip, sag_embs_flip = self.sag_model(x_flip, cond)
        ys2_flip = ys2_flip.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        # sag_embs: bs, 5_cond, 5_level, 128
        sag_embs = sag_embs.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128
        sag_embs_flip = sag_embs_flip.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128

        #
        embs = torch.cat((sag_embs, axial_embs), dim=-1)
        out1 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        out1 = out1.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        out1 = out1 + ys2

        # #
        # embs = torch.cat((sag_embs_flip, axial_embs), dim=-1)
        # out2 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        # out2 = out2.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        # out2 = out2 + ys2_flip
        #
        # #
        # embs = torch.cat((sag_embs, axial_embs_flip), dim=-1)
        # out3 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        # out3 = out3.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        # out3 = out3 + ys2

        #
        embs = torch.cat((sag_embs_flip, axial_embs_flip), dim=-1)
        out4 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        out4 = out4.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        out4 = out4 + ys2_flip

        # ys = (out1 + out2 + out3 + out4) / 4.0
        ys = (out1 + out4) / 2.0
        ys = ys.reshape(b, -1)
        return ys


from patriot.metric import CALC_score


def get_level_target(valid_df, level=0):
    names = ["spinal_canal_stenosis",
             "left_neural_foraminal_narrowing",
             "right_neural_foraminal_narrowing",
             "left_subarticular_stenosis",
             "right_subarticular_stenosis"
             ]
    _idx_to_level_name = {
        1: 'l1_l2',
        2: 'l2_l3',
        3: 'l3_l4',
        4: 'l4_l5',
        5: 'l5_s1',
    }
    for i in range(len(names)):
        names[i] = '{}_{}'.format(names[i], _idx_to_level_name[level])
    return valid_df[names].values


def make_calc(test_stusy, l5, l4, l3, l2, l1, folds_tmp):
    # ref from patriot
    new_df = pd.DataFrame()
    tra_df = list(pd.read_csv(f"{data_root}/train.csv").columns[1:])
    col = []
    c_ = []
    level = []
    for i in test_stusy:
        for j in tra_df:
            col.append(f"{i}_{j}")
            c_.append(f"{i}")
            level.append(j.split("_")[-2])

    # print(level[:10])

    new_df["row_id"] = col
    new_df["study_id"] = c_
    new_df["level"] = level

    new_df["level"] = new_df["level"].astype("str")
    new_df["row_id"] = new_df["row_id"].astype("str")
    new_df["normal_mild"] = 0
    new_df["moderate"] = 0
    new_df["severe"] = 0
    new_df___ = []
    name__2 = {"l5": 0, "l4": 1, "l3": 2, "l2": 3, "l1": 4}
    for pred, level in zip([l5, l4, l3, l2, l1], [5, 4, 3, 2, 1]):
        name_ = f'l{level}'
        new_df_ = new_df[new_df["level"] == name_]
        # fold_tmp_ = folds_tmp[folds_tmp["level"] == name__2[name_]][
        #     ["spinal_canal_stenosis", "left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing",
        #      "left_subarticular_stenosis", "right_subarticular_stenosis"]].values

        fold_tmp_ = get_level_target(folds_tmp, level)
        new_df_[["normal_mild", "moderate", "severe"]] = pred.reshape(-1, 3)
        new_df_["GT"] = fold_tmp_.reshape(-1, )
        new_df___.append(new_df_)

    new_df = pd.concat(new_df___).sort_values("row_id").reset_index(drop=True)

    new_df = new_df[new_df["GT"] != -100].reset_index(drop=True)
    # この時点でauc計算でも良い？
    GT = new_df.iloc[:, [0, -1]].copy()
    GT[["normal_mild", "moderate", "severe"]] = np.eye(3)[GT["GT"].to_numpy().astype(np.uint8)]
    GT["sample_weight"] = 2 ** GT["GT"].to_numpy()

    GT = GT.iloc[:, [0, 2, 3, 4, 5]]
    metirc_ = CALC_score(GT, new_df.iloc[:, [0, 3, 4, 5]], row_id_column_name="row_id")
    return metirc_, new_df

from torch.nn.modules.loss import _Loss


class SevereLoss(_Loss):
    """
    For RSNA 2024
    criterion = SevereLoss()     # you can replace nn.CrossEntropyLoss
    loss = criterion(y_pred, y)
    """

    def __init__(self, temperature=0.0):
        """
        Use max if temperature = 0
        """
        super().__init__()
        self.t = temperature
        assert self.t >= 0

    def __repr__(self):
        return 'SevereLoss(t=%.1f)' % self.t

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y_pred (Tensor[float]): logit             (batch_size, 3, 25)
          y      (Tensor[int]):   true label index  (batch_size, 25)
        """
        assert y_pred.size(0) == y.size(0)
        assert y_pred.size(1) == 3 and y_pred.size(2) == 25
        assert y.size(1) == 25
        assert y.size(0) > 0

        slices = [slice(0, 5), slice(5, 15), slice(15, 25)]
        w = 2 ** y  # sample_weight w = (1, 2, 4) for y = 0, 1, 2 (batch_size, 25)

        loss = F.cross_entropy(y_pred, y, reduction='none')  # (batch_size, 25)

        # Weighted sum of losses for spinal (:5), foraminal (5:15), and subarticular (15:25)
        wloss_sums = []
        for k, idx in enumerate(slices):
            wloss_sums.append((w[:, idx] * loss[:, idx]).sum())

        # any_severe_spinal
        #   True label y_max:      Is any of 5 spinal severe? true/false
        #   Prediction y_pred_max: Max of 5 spinal severe probabilities y_pred[:, 2, :5].max(dim=1)
        #   any_severe_spinal_loss is the binary cross entropy between these two.
        y_spinal_prob = y_pred[:, :, :5].softmax(dim=1)  # (batch_size, 3,  5)
        w_max = torch.amax(w[:, :5], dim=1)  # (batch_size, )
        # y_max = torch.amax(y[:, :5] == 2, dim=1).to(torch.float32)  # 0 or 1
        y_max = torch.amax(y[:, :5] == 2, dim=1).to(y_pred.dtype)  # 0 or 1

        if self.t > 0:
            # Attention for the maximum value
            attn = F.softmax(y_spinal_prob[:, 2, :] / self.t, dim=1)  # (batch_size, 5)

            # Pick the sofmax among 5 severe=2 y_spinal_probs with attn
            y_pred_max = (attn * y_spinal_prob[:, 2, :]).sum(dim=1)  # weighted average among 5 spinal columns
        else:
            # Exact max; this works too
            y_pred_max = y_spinal_prob[:, 2, :].amax(dim=1)

        loss_max = F.binary_cross_entropy(y_pred_max, y_max, reduction='none')
        wloss_sums.append((w_max * loss_max).sum())

        # See below about these numbers
        loss = (wloss_sums[0] / 6.084050632911392 +
                wloss_sums[1] / 12.962531645569621 +
                wloss_sums[2] / 14.38632911392405 +
                wloss_sums[3] / 1.729113924050633) / (4 * y.size(0))

        return loss


class SevereLoss_1_Level(_Loss):
    """
    For RSNA 2024
    criterion = SevereLoss()     # you can replace nn.CrossEntropyLoss
    loss = criterion(y_pred, y)
    """

    def __init__(self, temperature=0.0):
        """
        Use max if temperature = 0
        """
        super().__init__()
        self.t = temperature
        assert self.t >= 0

    def __repr__(self):
        return 'SevereLoss(t=%.1f)' % self.t

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
          y_pred (Tensor[float]): logit             (batch_size, 3, 5)
          y      (Tensor[int]):   true label index  (batch_size, 5)
        """
        assert y_pred.size(0) == y.size(0)
        assert y_pred.size(1) == 3 and y_pred.size(2) == 5
        assert y.size(1) == 5
        assert y.size(0) > 0

        slices = [slice(0, 1), slice(1, 3), slice(3, 5)]
        w = 2 ** y  # sample_weight w = (1, 2, 4) for y = 0, 1, 2 (batch_size, 5)

        loss = F.cross_entropy(y_pred, y, reduction='none')  # (batch_size, 25)

        # Weighted sum of losses for spinal (:5), foraminal (5:15), and subarticular (15:25)
        wloss_sums = []
        for k, idx in enumerate(slices):
            wloss_sums.append((w[:, idx] * loss[:, idx]).sum())

        # any_severe_spinal
        #   True label y_max:      Is any of 5 spinal severe? true/false
        #   Prediction y_pred_max: Max of 5 spinal severe probabilities y_pred[:, 2, :5].max(dim=1)
        #   any_severe_spinal_loss is the binary cross entropy between these two.
        # y_spinal_prob = y_pred[:, :, :1].softmax(dim=1)  # (batch_size, 3,  5)
        # w_max = torch.amax(w[:, :1], dim=1)  # (batch_size, )
        # # y_max = torch.amax(y[:, :5] == 2, dim=1).to(torch.float32)  # 0 or 1
        # y_max = torch.amax(y[:, :1] == 2, dim=1).to(y_pred.dtype)  # 0 or 1
        #
        # if self.t > 0:
        #     # Attention for the maximum value
        #     attn = F.softmax(y_spinal_prob[:, 2, :] / self.t, dim=1)  # (batch_size, 5)
        #
        #     # Pick the sofmax among 5 severe=2 y_spinal_probs with attn
        #     y_pred_max = (attn * y_spinal_prob[:, 2, :]).sum(dim=1)  # weighted average among 5 spinal columns
        # else:
        #     # Exact max; this works too
        #     y_pred_max = y_spinal_prob[:, 2, :].amax(dim=1)
        #
        # loss_max = F.cross_entropy(y_pred_max, y_max, reduction='none')
        # wloss_sums.append((w_max * loss_max).sum())

        # See below about these numbers
        loss = (wloss_sums[0] / 6.084050632911392 +
                wloss_sums[1] / 12.962531645569621 +
                wloss_sums[2] / 14.38632911392405# +
                #wloss_sums[3] / 1.729113924050633
                ) / (4 * y.size(0))

        return loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_sag', type=str, default='convnext_small.in12k_ft_in1k_384')
    parser.add_argument('--backbone_axial', type=str, default='densenet201')
    parser.add_argument('--save_dir', type=str, default='./wkdir/cls/v2_hybrid_exp/')
    parser.add_argument('--only_val', type=int, default=0)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--axial_crop_with_model', type=int, default=0)
    parser.add_argument('--axial_crop_size', type=int, default=300)
    parser.add_argument('--sag_img_size', type=int, default=128)
    parser.add_argument('--axial_margin_extend', type=float, default=1.0)
    parser.add_argument('--bs', type=int, default=32)
    return parser.parse_args()


args = get_args()

axial_crop_with_model = True if args.axial_crop_with_model == 1 else False
axial_crop_size = args.axial_crop_size
sag_img_size = args.sag_img_size
###
EXP_NUM = 624
NOT_DEBUG = True

OUTPUT_DIR = f'{args.save_dir}/{args.backbone_sag}_axial_{args.backbone_axial}_' \
             f'axial_size_{axial_crop_size}_sag_size_{sag_img_size}'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
N_WORKERS = 12
USE_AMP = True
SEED = 8620

IMG_SIZE = [128, 128]
IN_CHANS = 3
N_LABELS = 25  # 15
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
#data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'

data_root = '/home/hw/ssd_01_new/kaggle/rsna-2024-lumbar-spine-degenerative-classification/'
# data_root = '/root/autodl-tmp/data/'
# volume_data_root = f'{data_root}/train_images_preprocessed/'
df = pd.read_csv(f'{data_root}/train.csv')

# # Find rows with NaN values
# nan_rows = df[df.isna().any(axis=1)]

# # Print rows with NaN values
# print(nan_rows.iloc[0])
# exit(0)

df = df.fillna(-100)
label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
df = df.replace(label2id)

# without = [
#             490052995,
#             1261271580,
#             2507107985,
#             2626030939,
#             2773343225,
#             3109648055,
#             3387993595,
#             2492114990, 3008676218, 2780132468, 3637444890
#         ]
# hard_axial_study_id_list = [391103067,
#                                     953639220,
#                                     2460381798,
#                                     2690161683,
#                                     3650821463,
#                                     3949892272,
#                                     677672203,  # 左右点标注反了
#                                     ]
#
# df = df[~df['study_id'].isin(hard_axial_study_id_list)]

aux_info = get_train_study_aux_info(data_root)

###
transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=AUG_PROB),
    # A.OneOf([
    #     #A.OpticalDistortion(distort_limit=1.0),
    #     # A.GridDistortion(num_steps=5, distort_limit=1.),
    #     A.ElasticTransform(alpha=3),
    # ], p=AUG_PROB),
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

if axial_crop_with_model:
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

else:
    transforms_train_axial = A.Compose([
        # A.Resize(512, 512),
        A.CenterCrop(axial_crop_size, axial_crop_size),
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
        # A.Resize(512, 512),
        A.CenterCrop(axial_crop_size, axial_crop_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

if not NOT_DEBUG or not AUG:
    transforms_train = transforms_val


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


def mixup_data(x1, x2, t, alpha=1.0, use_cuda=True):
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
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :, :]

    t_a, t_b = t, t[index]
    return mixed_x1, mixed_x2, t_a, t_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def collate_fn(batch):
    img = []
    axial_imgs = []
    label = []
    cond = []

    for item in batch:
        img.append(item['img'])
        axial_imgs.append(item['axial_imgs'])
        label.append(item['label'])
        cond.append(item['cond'])

    return {'img': torch.stack(img, dim=0),
            'axial_imgs': axial_imgs, #torch.stack(axial_imgs, dim=0),#axial_imgs,
            'label': torch.stack(label, dim=0),
            'cond': torch.stack(cond, dim=0),
            }

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer

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

        model = HybridModel(args.backbone_sag,
                            args.backbone_axial, pretrained=True)
        # fname = f'../patriot//fold{fold}_base_00b_3ch_v2s_ccrop08_best_metric.pth'
        # model.axial_model.load_state_dict(torch.load(fname), strict=False)

        model.to(device)
        # exit(0)

        train_ds = RSNA24Dataset_Cls_V2(
            data_root,
            aux_info,
            df_train, phase='train', transform=transforms_train,
            transforms_axial=transforms_train_axial,
            level_by_level=True,
            axial_crop_xy_size=axial_crop_size if axial_crop_with_model else -1,
            axial_margin_extend=args.axial_margin_extend,
            sag_img_size=args.sag_img_size
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            # pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS,
        )

        # valid_ds = RSNA24Dataset_Cls_V2(
        #     data_root,
        #     aux_info,
        #     df_valid, phase='valid', transform=transforms_val,
        #     transforms_axial=transforms_val_axial,
        #     axial_crop_xy_size=axial_crop_size if axial_crop_with_model else -1,
        #     axial_margin_extend=args.axial_margin_extend,
        # )
        # valid_dl = DataLoader(
        #     valid_ds,
        #     batch_size=8,
        #     shuffle=False,
        #     # pin_memory=True,
        #     drop_last=False,
        #     num_workers=N_WORKERS
        # )
        valid_ds = RSNA24Dataset_Cls_V2_Multi_Axial(data_root, aux_info,
                                                    df_valid, phase='valid', transform=transforms_val,
                                                    transforms_axial=transforms_val_axial,
                                                    level_by_level=False,
                                                    axial_crop_xy_size=axial_crop_size if axial_crop_with_model else -1,
                                                    axial_margin_extend=args.axial_margin_extend,
                                                    sag_img_size=args.sag_img_size)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=4,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=N_WORKERS,
            collate_fn=collate_fn
        )

        # refine_ds = RSNA24Dataset_Cls_V2(
        #     data_root,
        #     aux_info,
        #     df_train, phase='refine', transform=transforms_val,
        #     transforms_axial=transforms_val_axial,
        #     level_by_level=True,
        #     axial_crop_xy_size=axial_crop_size if axial_crop_with_model else -1,
        #     axial_margin_extend=args.axial_margin_extend,
        # )
        # refine_dl = DataLoader(
        #     refine_ds,
        #     batch_size=BATCH_SIZE,
        #     shuffle=True,
        #     # pin_memory=True,
        #     drop_last=True,
        #     num_workers=N_WORKERS
        # )

        if fold == 0:
            print('has:', sum(p.numel() for p in model.parameters()), 'params')

        model_ema = ModelEmaV2(model, decay=0.99)

        # for param in model.parameters():
        #     param.requires_grad = False
        #
        # layers_to_optimize = [model.sag_model, model.out_linear]
        # for layer in layers_to_optimize:
        #     for param in layer.parameters():
        #         param.requires_grad = True

        model_params = get_params(model, LR, decay=WD)

        optimizer = AdamW(model_params, lr=LR)

        warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
        num_total_steps = (EPOCHS + 3) * len(train_dl) // GRAD_ACC
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2, )

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        severeLoss = SevereLoss_1_Level(temperature=0)

        best_loss = 1.1
        best_loss_ema = 1.1
        best_metric = 1.2
        best_metric_ema = 1.2
        best_score = 1.2
        es_step = 0

        for epoch in range(1, EPOCHS + 4):
            if epoch == EPOCHS + 1:
                fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'  # if best_metric<best_metric_ema else f'{OUTPUT_DIR}/best_wll_model_fold-{fold}_ema.pt'
                model.load_state_dict(torch.load(fname))
                model.to(device)

            logger.write(f'start epoch {epoch}')
            model.train()
            total_loss = 0
            #with tqdm(train_dl if epoch <= EPOCHS else refine_dl, leave=True) as pbar:
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['img']
                    t = tensor_dict['label']
                    cond = tensor_dict['cond']
                    x = x.to(device)
                    t = t.to(device)
                    cond = cond.to(device)
                    axial_x = tensor_dict['axial_imgs']
                    axial_x = axial_x.to(device)

                    with autocast:
                        loss = 0
                        do_mixup = False
                        do_mixup = np.random.rand() < 0.2
                        if epoch > 6:
                            do_mixup = False
                        
                        if do_mixup:
                            x, axial_x, t_a, t_b, lam = mixup_data(x, axial_x, t)

                        y = model(x, axial_x, cond)
                        N_cond = 5
                        for col in range(N_cond):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
                            if do_mixup:
                                gt_a = t_a[:, col]
                                gt_b = t_b[:, col]
                                loss_part = mixup_criterion(criterion, pred, gt_a, gt_b, lam) / N_cond
                            else:
                                loss_part = criterion(pred, gt) / N_cond
                            loss = loss + loss_part

#                         bs = x.shape[0]
#                         y = y.reshape(bs, 5, 3).permute(0, 2, 1)
#                         loss = severeLoss(y, t)

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
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, tensor_dict in enumerate(pbar):
                        x = tensor_dict['img']
                        t = tensor_dict['label']
                        cond = tensor_dict['cond']
                        x = x.to(device)
                        t = t.to(device)
                        cond = cond.to(device)
                        axial_x = tensor_dict['axial_imgs']
                        for j in range(len(axial_x)):
                            axial_x[j] = axial_x[j].to(device)
                        # axial_x = axial_x.to(device)

                        with autocast:
                            loss = 0
                            loss_ema = 0
                            y = model.forward_test_tta_axial_dynamic(x, axial_x, cond)

                            for col in range(N_LABELS):
                                pred = y[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()

                            total_loss += loss.item()

                            y_ema = model_ema.module.forward_test_tta_axial_dynamic(x, axial_x, cond)
                            for col in range(N_LABELS):
                                pred = y_ema[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                loss_ema = loss_ema + criterion(pred, gt) / N_LABELS
                                y_pred_ema = pred.float()
                                y_preds_ema.append(y_pred_ema.cpu())

                            bs, _ = y_ema.shape
                            # bs, 5_cond, 5_level, 3
                            y_ema = y_ema.reshape(bs, 5, 5, 3)
                            fold_preds.append(y_ema)

            fold_preds = torch.cat(fold_preds, dim=0)
            fold_preds = nn.Softmax(dim=-1)(fold_preds).cpu().numpy()
            l5 = fold_preds[:, :, 4, :]
            l4 = fold_preds[:, :, 3, :]
            l3 = fold_preds[:, :, 2, :]
            l2 = fold_preds[:, :, 1, :]
            l1 = fold_preds[:, :, 0, :]

            c, val_df_ = make_calc(df_valid["study_id"].unique(),
                                   l5, l4, l3, l2, l1, df_valid)
            logger.write(f"metric  {c}")
            

            y_preds = torch.cat(y_preds, dim=0)
            y_preds_ema = torch.cat(y_preds_ema, dim=0)
            labels = torch.cat(labels)

            met = criterion2(y_preds, labels)
            met_ema = criterion2(y_preds_ema, labels)

            val_loss = total_loss / len(valid_dl)
            # if scheduler is not None:
            #     scheduler.step(val_loss)

            logger.write(f'val_loss:{val_loss:.6f}, val_Metric:{met:.6f}, val_Metric_ema:{met_ema:.6f}')
            #if val_loss < best_loss or met < best_metric or met_ema < best_metric_ema:
            if c < best_score:
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
                    #torch.save(model_ema.module.state_dict(), fname)

                if c < best_score:
                    logger.write(f'epoch:{epoch}, best score updated from {best_score:.6f} to {c:.6f}')
                    best_score = c
                    fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}_score.pt'
                    torch.save(model_ema.module.state_dict(), fname)
                if device != 'cuda:0':
                    model.to(device)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    logger.write('early stopping')
                    break

        logger.write('best score: {}\n'.format(best_score))
        logger.write('###' * 10)
        # assert best_metric < 0.75
        # exit(0)


#easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()

#### val
cv = 0
y_preds = []
y_preds_ema = []
labels = []
weights = torch.tensor([1.0, 2.0, 4.0])
criterion2 = nn.CrossEntropyLoss(weight=weights)
metric_scores = []
for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
    # if fold !=4:
    #     continue
    print('#' * 30)
    print(f'start fold{fold}')
    print('#' * 30)
    # df_valid = df.iloc[val_idx]

    folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
    val_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index
    folds_t2s_each = folds_t2s_each.loc[val_idx].copy().reset_index(drop=True)
    df_valid = df[df['study_id'].isin(folds_t2s_each['study_id'])].copy().reset_index(drop=True)
    # df_valid = df_valid.iloc[:10]

    #df_valid = df_valid[df_valid['study_id'].isin(easy_study_ids)]

    valid_ds = RSNA24Dataset_Cls_V2_Multi_Axial(data_root, aux_info,
                                    df_valid, phase='valid', transform=transforms_val,
                                    transforms_axial=transforms_val_axial,
                                    level_by_level=False,
                                    axial_crop_xy_size=axial_crop_size if axial_crop_with_model else -1,
                                    axial_margin_extend=args.axial_margin_extend,
                                    sag_img_size=args.sag_img_size)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=N_WORKERS,
        collate_fn=collate_fn
    )
    modele = HybridModel(args.backbone_sag,
                         args.backbone_axial, pretrained=True)
    fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}_score.pt'
    modele.load_state_dict(torch.load(fname))
    modele.to(device)

    # model.eval()
    modele.eval()
    # modele = modele.half()
    fold_preds = []
    with tqdm(valid_dl, leave=True) as pbar:
        with torch.no_grad():
            for idx, tensor_dict in enumerate(pbar):
                x = tensor_dict['img']
                t = tensor_dict['label']
                cond = tensor_dict['cond']
                x = x.to(device)
                t = t.to(device)
                cond = cond.to(device)
                axial_x = tensor_dict['axial_imgs']
                for j in range(len(axial_x)):
                    axial_x[j] = axial_x[j].to(device)
                #axial_x = axial_x.to(device)

                with autocast:
                    # y = model(x,axial_x, cond)
                    ye = modele.forward_test_tta_axial_dynamic(x, axial_x, cond)

                    for col in range(0, N_LABELS):
                        # pred = y[:, col * 3:col * 3 + 3]
                        prede = ye[:, col * 3:col * 3 + 3]
                        gt = t[:, col]
                        # y_pred = pred.float()
                        # y_preds.append(y_pred.cpu())
                        y_preds_ema.append(prede.float().cpu())
                        labels.append(gt.cpu())
                    bs, _ = ye.shape
                    # bs, 5_cond, 5_level, 3
                    ye = ye.reshape(bs, 5, 5, 3)
                    fold_preds.append(ye)

    fold_preds = torch.cat(fold_preds, dim=0)
    fold_preds = nn.Softmax(dim=-1)(fold_preds).cpu().numpy()
    l5 = fold_preds[:, :, 4, :]
    l4 = fold_preds[:, :, 3, :]
    l3 = fold_preds[:, :, 2, :]
    l2 = fold_preds[:, :, 1, :]
    l1 = fold_preds[:, :, 0, :]

    c, val_df_ = make_calc(df_valid["study_id"].unique(),
                           l5, l4, l3, l2, l1, df_valid)
    print(f"metric  {c}")
    metric_scores.append(c)


logger.write('metric scores: {}'.format(metric_scores))
logger.write('metric avg scores: {}'.format(np.mean(metric_scores)))

# y_preds = torch.cat(y_preds)
y_preds_ema = torch.cat(y_preds_ema)
labels = torch.cat(labels)

# cv = criterion2(y_preds, labels)
# logger.write('cv score: {}'.format(cv.item()))

cv = criterion2(y_preds_ema, labels)
logger.write('cv score ema: {}'.format(cv.item()))

from sklearn.metrics import log_loss

# y_pred_np = y_preds.softmax(1).numpy()
y_prede_np = y_preds_ema.softmax(1).numpy()

labels_np = labels.numpy()
y_pred_nan = np.zeros((y_preds_ema.shape[0], 1))
# y_pred2 = np.concatenate([y_pred_nan, y_pred_np], axis=1)
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
logger.write('cv score sklearn: {}'.format(cv2))

np.save(f'{OUTPUT_DIR}/labels.npy', labels_np)
np.save(f'{OUTPUT_DIR}/final_pred_ema.npy', y_prede)

# np.save(f'{OUTPUT_DIR}/final_{EXP_NUM:03d}_{MODEL_NAME.split(".")[0]}.npy', y_pred2)
# np.save(f'{OUTPUT_DIR}/final_{EXP_NUM:03d}_{MODEL_NAME.split(".")[0]}_ema.npy', y_prede)
