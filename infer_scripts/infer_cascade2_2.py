# -*- coding: utf-8 -*-
import os
import pickle

import timm_3d

os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import KFold

import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import timm
import random

import glob
import pydicom
from skimage.transform import resize

from sklearn.cluster import KMeans
from src.utils.comm import setup_seed
import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


#### keypoint model
class RSNA24Model_Keypoint_3D_Axial(nn.Module):
    def __init__(self, model_name='densenet161', pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=1,
            num_classes=30,
            global_pool='avg'
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x


class RSNA24Model_Keypoint_3D_Sag(nn.Module):
    def __init__(self, model_name='densenet161', pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=2,
            num_classes=45,
            global_pool='avg'
        )

    def forward(self, x):
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x


#### cls  model
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
            embeds = embeds.reshape(b, 5, -1)  # b, 5, n_cond*128

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        if with_emb:
            return x, embeds
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
        # print('x shape: ', x.shape)
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
        self.sag_t1_model = base_model_cls(model_name, in_chans=in_chans,
                                           pretrained=pretrained)
        self.sag_t2_model = base_model_cls(model_name, in_chans=in_chans,
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


### Axial
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


class AxialModel_Fuse(nn.Module):
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
        # return self.forward_train(x, axial_x, cond)
        if self.training:
            return self.forward_train(x, axial_x, cond)
        else:
            return self.forward_test(x, axial_x, cond)

    def forward_train(self, x, axial_x, cond):
        sag_pred, sag_emb = self.sag_model.forward(x, cond, with_emb=True)
        bs = axial_x.shape[0]
        # bs, 3_cond, 5_level, 3
        # sag_pred = sag_pred.reshape(bs, 5, 5, 3)
        # sag_pred = sag_pred[:, 1:3, :, :]

        axial_pred, axial_emb = self.axial_model.forward(axial_x, with_emb=True)
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

    def forward_test(self, x, axial_x, cond):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        # axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),
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


class HybridModel(nn.Module):
    def __init__(self, backbone_sag='convnext_small.in12k_ft_in1k_384',
                 backbone_axial='convnext_small.in12k_ft_in1k_384',
                 pretrained=False):
        super().__init__()
        self.sag_model = Sag_Model_25D(backbone_sag, pretrained=pretrained)
        self.axial_model = Axial_Model_25D_GRU(base_model=backbone_axial,
                                               pretrained=pretrained)
        fdim = 512 + 5 * 128
        self.out_linear = nn.Linear(fdim, 15)

    def forward(self, x, axial_x, cond):
        # return self.forward_train(x, axial_x, cond)
        if self.training:
            return self.forward_dynamic(x, axial_x, cond)
        else:
            return self.forward_test(x, axial_x, cond)

    def forward_test(self, x, axial_x, cond):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        # axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),
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
        sag_pred = sag_pred.reshape(bs, 5, 5, 3)

        axial_pred, axial_emb = [], []
        for axial_x in axial_x_list:
            axial_x = axial_x.unsqueeze(0)
            _, n_series, c, h, w = axial_x.shape
            n_series = n_series // 10
            axial_x = axial_x.reshape(n_series, 10, c, h, w)
            p, emb = self.axial_model.forward(axial_x, with_emb=True)
            axial_pred.append(p.mean(dim=0, keepdims=True))
            axial_emb.append(emb.mean(dim=0, keepdims=True))

        # axial_pred = torch.cat(axial_pred, dim=0)
        axial_emb = torch.cat(axial_emb, dim=0)

        # bs, 5_cond, 5_level, 3
        ys = sag_pred

        ### fuse ####
        # bs, 5_level, fdim
        fea = torch.cat((sag_emb, axial_emb), dim=-1)
        # bs, 5_level, 5_cond, 3
        ys2 = self.out_linear(fea).reshape(bs, 5, 5, 3)
        # bs, 5_cond, 5_level, 3
        ys2 = ys2.permute(0, 2, 1, 3)
        # saggital keep independent
        # ys[:, 3: ] = (ys[:, 3: ] + ys2[:, 3: ]) / 2
        ys = (ys + ys2) / 2

        ys = ys.reshape(bs, -1)
        return ys


class Ensemble_Model(nn.Module):
    def __init__(self,
                 model_name1='convnext_small.in12k_ft_in1k_384',
                 model_name2='convnext_small.in12k_ft_in1k_384',
                 model_name3='convnext_small.in12k_ft_in1k_384',
                 pretrained=False):
        super(Ensemble_Model, self).__init__()
        # self.sag_t1_model = Sag_Model_25D(model_name1, pretrained=pretrained)
        # self.sag_t2_model = Sag_Model_25D(model_name2, pretrained=pretrained)

        self.sag_model = Sag_Model_T1_T2(model_name1, base_model_cls=Sag_Model_25D_GRU)

        self.axial_t2_model = AxialModel_Fuse(backbone_sag=model_name3,
                                              backbone_axial=model_name3,
                                              pretrained=pretrained)
        self.hybrid_model = HybridModel()

    def forward(self, x, axial_x, cond, n_list):
        bs = x.shape[0]
        axial_x_list = []
        for i in range(bs):
            axial_x_list.append(axial_x[i, :n_list[i]])

        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        pred = self.sag_model(x, cond)
        pred_flip = self.sag_model(x_flip, cond)
        sag_pred = (pred + pred_flip) / 2

        axial_pred = self.axial_t2_model(x, axial_x_list, cond)
        pred = torch.cat((sag_pred, axial_pred), dim=-1)

        pred2 = self.hybrid_model(x, axial_x_list, cond)
        pred = 0.6 * pred + 0.4 * pred2

        return pred


####
# lhwcv dataset related
G_IMG_SIZE = 512

def get_test_des_to_sids(data_root, test_series_descriptions_fn):
    test_series = pd.read_csv(
        f'{data_root}/{test_series_descriptions_fn}')

    infos = {}
    for study_id in test_series['study_id'].unique().tolist():
        study_series = test_series[test_series['study_id'] == study_id]
        des_to_sid = {}
        for sid, des in zip(study_series['series_id'], study_series['series_description']):
            if des in des_to_sid.keys():
                des_to_sid[des].append(sid)
            else:
                des_to_sid[des] = [sid]
        infos[study_id] = {
            'des_to_sid': des_to_sid
        }
    return infos


def convert_to_8bit_lhw_version(x):
    lower, upper = np.percentile(x, (1, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
    return x.astype("uint8")


def resize_volume(x, w=640, h=640):
    xs = []
    for i in range(len(x)):
        img = x[i]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        xs.append(img)
    return np.array(xs, dtype=np.uint8)


def load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    if len(dicom_files) == 0:
        return {
            'array': None
        }
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicom_instance_numbers = [int(i.split('/')[-1][:-4]) for i in dicom_files]

    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    dicom_instance_numbers = np.array(dicom_instance_numbers)[idx]

    cols = np.asarray([d.pixel_array.shape[1] for d in dicoms]).astype("int")[idx].tolist()
    rows = np.asarray([d.pixel_array.shape[0] for d in dicoms]).astype("int")[idx].tolist()
    instance_num_to_shape = {}
    dicom_instance_numbers = dicom_instance_numbers.tolist()
    for i, n in enumerate(dicom_instance_numbers):
        instance_num_to_shape[n] = [cols[i], rows[i]]

    array = []
    for i, d in enumerate(dicoms):
        arr = d.pixel_array.astype("float32")
        arr = resize(arr, (512, 512))
        array.append(arr)
    array = np.array(array)
    array = array[idx]
    array = convert_to_8bit_lhw_version(array)
    # array = resize_volume(array, 512, 512)

    return {"array": array, }


def gen_level_group_idxs(pred_z_list, depth, ext=1.0):
    z1, z2, z3, z4, z5 = pred_z_list
    margin1 = ext * abs(z2 - z1) / 2
    z1_start = z1 - margin1
    z1_end = z1 + margin1

    margin2 = ext * abs(z3 - z2) / 2
    z2_start = z2 - margin1
    z2_end = z2 + margin2

    margin3 = ext * abs(z4 - z3) / 2
    z3_start = z3 - margin2
    z3_end = z3 + margin3

    margin4 = ext * abs(z5 - z4) / 2
    z4_start = z4 - margin3
    z4_end = z4 + margin4

    z5_start = z5 - margin4
    z5_end = z5 + margin4

    group_idxs = np.array([
        [z1_start, z1_end],
        [z2_start, z2_end],
        [z3_start, z3_end],
        [z4_start, z4_end],
        [z5_start, z5_end],
    ])
    group_idxs = np.clip(group_idxs, 0, depth - 1)
    group_idxs = np.round(group_idxs)
    return group_idxs


def build_axial_group_idxs(pred_keypoints_infos0, img_size=512):
    dict_axial_group_idxs = {}

    # pred_keypoints_infos0 = pickle.load(
    #     open(f'{data_root}/v2_axial_3d_keypoints_model0.pkl', 'rb'))

    for study_id in pred_keypoints_infos0.keys():
        for sid in pred_keypoints_infos0[study_id].keys():
            pred_info0 = pred_keypoints_infos0[study_id][sid]
            predict_keypoints = pred_info0['points']
            depth = int(pred_info0['d'])

            scale = img_size / 4.0
            scale_z = depth / 16.0
            predict_keypoints[:, :2] = predict_keypoints[:, :2] * scale
            predict_keypoints[:, 2] = predict_keypoints[:, 2] * scale_z

            pred_z = predict_keypoints[:, 2].reshape(2, 5)
            pred_z = np.round(pred_z)
            group_idxs = gen_level_group_idxs(pred_z.mean(axis=0), depth, 1.0)
            group_idxs = np.asarray(group_idxs, dtype=np.int64)

            xy = predict_keypoints[:, :2].reshape(2, 5, 2)
            center_xy = np.round(xy.mean(axis=0))

            if study_id not in dict_axial_group_idxs.keys():
                dict_axial_group_idxs[study_id] = {}
            dict_axial_group_idxs[study_id][sid] = {
                'group_idxs': group_idxs,
                'center_xy': center_xy,
                'pred_z': pred_z,
                'pred_xy': xy,
                'n_slices': depth
            }
    return dict_axial_group_idxs


class RSNA24DatasetTest_LHW_keypoint_3D_Saggital(Dataset):
    def __init__(self,
                 data_root,
                 test_series_descriptions_fn,
                 study_ids,
                 transform=None,
                 image_dir=None):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root, test_series_descriptions_fn)

        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

    def __len__(self):
        return len(self.study_ids)

    def get_3d_volume(self, study_id, sid, series_description):
        # TODO use multi when test
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))
        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            plane = "axial"
            reverse_sort = True
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']
        return volume, sid

    def __getitem__(self, idx):

        study_id = self.study_ids[idx]
        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        has_t1 = True
        if 'Sagittal T1' not in des_to_sid.keys():
            has_t1 = False
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
        else:
            # TODO use multi
            sid = des_to_sid['Sagittal T1'][0]
            s_t1,_ = self.get_3d_volume(study_id, sid, "Sagittal T1")

        # Sagittal T2/STIR
        has_t2 = True
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            has_t2 = False
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
        else:
            # TODO use multi
            sid = des_to_sid['Sagittal T2/STIR'][0]
            s_t2, _ = self.get_3d_volume(study_id, sid, "Sagittal T2/STIR")

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t1 = self.transform(image=s_t1)['image']
            s_t1 = s_t1.transpose(2, 0, 1)

            s_t2 = s_t2.transpose(1, 2, 0)
            s_t2 = self.transform(image=s_t2)['image']
            s_t2 = s_t2.transpose(2, 0, 1)

        s_t1 = s_t1.astype(np.float32)
        s_t1 = torch.from_numpy(s_t1).unsqueeze(0).unsqueeze(0)
        s_t1 = F.interpolate(s_t1, size=(48, 256, 256)).squeeze(0)

        s_t2 = s_t2.astype(np.float32)
        s_t2 = torch.from_numpy(s_t2).unsqueeze(0).unsqueeze(0)
        s_t2 = F.interpolate(s_t2, size=(48, 256, 256)).squeeze(0)

        x = torch.cat((s_t1, s_t2), dim=0)

        return x, study_id


class RSNA24DatasetTest_LHW_keypoint_3D_Axial(Dataset):
    def __init__(self,
                 data_root,
                 test_series_descriptions_fn,
                 study_ids,
                 transform=None,
                 image_dir=None):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root, test_series_descriptions_fn)

        self.samples = []
        for study_id in study_ids:
            info = self.aux_info[study_id]
            sids = info['des_to_sid']['Axial T2']

            for sid in sids:
                self.samples.append({
                    'study_id': study_id,
                    'sid': sid,
                })
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform
        self.img_size = 256
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

    def __len__(self):
        return len(self.samples)

    def get_3d_volume(self, study_id, sid, series_description):
        # TODO use multi when test
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))
        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            plane = "axial"
            reverse_sort = True
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']
        return volume, sid

    def __getitem__(self, idx):
        a = self.samples[idx]
        study_id = a['study_id']
        sid = a['sid']

        # Axial T2
        a_t2, sid = self.get_3d_volume(study_id, sid, 'Axial T2')

        if self.transform is not None:
            # to h,w,c
            a_t2 = a_t2.transpose(1, 2, 0)
            a_t2 = self.transform(image=a_t2)['image']
            a_t2 = a_t2.transpose(2, 0, 1)

        x = a_t2
        x = x.astype(np.float32)
        depth = x.shape[0]
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(96, 256, 256)).squeeze()
        return x, study_id, sid, depth



class RSNA24DatasetTest_LHW_V2_2(Dataset):
    def __init__(self,
                 data_root,
                 test_series_descriptions_fn,
                 study_ids,
                 pred_sag_keypoints_infos_3d,
                 study_id_to_pred_keypoints_axial,
                 transform=None,
                 axial_transform=None,
                 gt_df_maybe=None,
                 image_dir=None,
                 ):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root, test_series_descriptions_fn)
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        self.transform = transform
        if axial_transform is None:
            axial_transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.axial_transform = axial_transform
        self.in_channels = 30
        self.img_size = 512
        self.gt_df_maybe = gt_df_maybe
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

        self.pred_sag_keypoints_infos_3d = pred_sag_keypoints_infos_3d
        self.dict_axial_group_idxs = build_axial_group_idxs(study_id_to_pred_keypoints_axial)
        self.z_imgs = 3

    def __len__(self):
        return len(self.study_ids)

    def slice_volume(self, volume, index, bbox=None):
        arr = []
        for i in index:
            a = volume[i]
            if bbox is not None:
                a = a[bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
                a = cv2.resize(a, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            arr.append(a)
        arr = np.array(arr, np.uint8)
        return arr

    def gen_slice_index_Sagittal(self, length):
        # step = length / 15.0
        # st = length / 2.0 - 6.5 * step
        # end = length + 0.0001
        step = length / 10.0
        st = length / 2.0 - 4.0 * step
        end = length + 0.0001
        slice_indexes = []
        for i in np.arange(st, end, step):
            inx = max(0, int((i - 0.5001).round()))
            slice_indexes.append(inx)
        return slice_indexes

    def gen_slice_index_Axial(self, length):
        # step = length / 15.0
        # st = length / 2.0 - 6.5 * step
        # end = length + 0.0001
        step = length / 10.0
        st = length / 2.0 - 4.0 * step
        end = length + 0.0001
        slice_indexes = []
        for i in np.arange(st, end, step):
            inx = max(0, int((i - 0.5001).round()))
            slice_indexes.append(inx)
        return slice_indexes

    # def load_axial(self, study_id, sid):
    #     return load_dicom_line_par(self.image_dir, study_id, sid)

    def get_3d_volume(self, study_id, sids, series_description):
        # TODO use multi when test
        sid = sids[0]
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))
        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            plane = "axial"
            reverse_sort = True
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']

        return volume, sid

    def crop_out_by_keypoints(self, volume, keypoints_xyz,
                              z_imgs=3, crop_size=128,
                              transform=None):
        sub_volume_list = []
        for p in keypoints_xyz:
            x, y, z = np.round(p[0]), np.round(p[1]), np.round(p[2])
            x, y, z = int(x), int(y), int(z)
            # no z
            if z < 0:
                v = np.zeros((z_imgs, crop_size, crop_size), dtype=volume.dtype)
                sub_volume_list.append(v)
                continue
            bbox = [x - crop_size // 2,
                    y - crop_size // 2,
                    x + crop_size // 2,
                    y + crop_size // 2]
            # 如果bbox在边界，偏移它以使其具有正确的img_size大小
            if bbox[0] < 0:
                bbox[2] = crop_size
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[3] = crop_size
                bbox[1] = 0
            if bbox[2] > G_IMG_SIZE:
                bbox[0] = G_IMG_SIZE - crop_size
                bbox[2] = G_IMG_SIZE
            if bbox[3] > G_IMG_SIZE:
                bbox[1] = G_IMG_SIZE - crop_size
                bbox[3] = G_IMG_SIZE

            bbox = [int(e) for e in bbox]
            z0 = z - z_imgs // 2
            z1 = z + z_imgs // 2 + 1
            if z0 < 0:
                z0 = 0
                z1 = z_imgs
            if z1 > volume.shape[0]:
                z0 = volume.shape[0] - z_imgs
                z1 = volume.shape[0]
            # print(z0, z1)
            v = volume[z0: z1, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()

            if transform is not None:
                v = v.transpose(1, 2, 0)
                v = transform(image=v)['image']
                v = v.transpose(2, 0, 1)

            sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        return volume_crop, None

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']
        # Sagittal T1
        if 'Sagittal T1' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
        else:
            s_t1, sid = self.get_3d_volume(study_id,
                                             des_to_sid['Sagittal T1'],
                                             'Sagittal T1')
            if s_t1 is None:
                s_t1 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)

        # Sagittal T2/STIR
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
        else:
            s_t2, sid = self.get_3d_volume(study_id,
                                             des_to_sid['Sagittal T2/STIR'],
                                             'Sagittal T2/STIR')
            if s_t2 is None:
                s_t2 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)

        c1 = s_t1.shape[0]
        c2 = s_t2.shape[0]

        # predicted keypoints
        pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id]['points'].reshape(15, 3)
        pred_keypoints[:, :2] = 512 * pred_keypoints[:, :2] / 4.0
        pred_keypoints[:5, 2] = c2 * pred_keypoints[:5, 2] / 16.0
        pred_keypoints[5:, 2] = c1 * pred_keypoints[5:, 2] / 16.0

        pred_keypoints = np.round(pred_keypoints).astype(np.int64)

        t2_keypoints = pred_keypoints[:5]
        t1_keypoints_left = pred_keypoints[5:10]
        t1_keypoints_right = pred_keypoints[10:15]

        keypoints_xy = np.concatenate((t2_keypoints[:, :2],
                                       t1_keypoints_left[:, :2],
                                       t1_keypoints_right[:, :2],
                                       ), axis=0)
        keypoints_z = np.concatenate((t2_keypoints[:, 2:],
                                      t1_keypoints_left[:, 2:],
                                      t1_keypoints_right[:, 2:],
                                      ), axis=0)
        mask_xy = np.where(keypoints_xy == -1, 0, 1)
        mask_z = np.where(keypoints_z == -1, 0, 1)

        if self.transform is not None:
            # same aug
            img = np.concatenate((s_t1, s_t2), axis=0)
            # to h,w,c
            img = img.transpose(1, 2, 0)
            augmented = self.transform(image=img,)
                                       #keypoints=keypoints_xy)
            img = augmented['image']
            # keypoints_xy = augmented['keypoints']
            # keypoints_xy = np.array(keypoints_xy)
            img = img.transpose(2, 0, 1)
            s_t1 = img[:c1]
            s_t2 = img[c1:]

        keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)
        mask = np.concatenate((mask_xy, mask_z), axis=1)

        s_t2_keypoints = keypoints[:5]
        s_t1_left_keypoints = keypoints[5:10]
        s_t1_right_keypoints = keypoints[10:15]

        crop_size = 128
        s_t2, _ = self.crop_out_by_keypoints(s_t2, s_t2_keypoints,
                                             z_imgs=self.z_imgs, crop_size=crop_size)

        s_t1_left, _ = self.crop_out_by_keypoints(s_t1, s_t1_left_keypoints,
                                                  z_imgs=self.z_imgs, crop_size=crop_size)
        #
        s_t1_right, _ = self.crop_out_by_keypoints(s_t1, s_t1_right_keypoints,
                                                   z_imgs=self.z_imgs, crop_size=crop_size)

        # s_t2 is also used to help axial
        x = np.concatenate([s_t2, s_t1_left, s_t1_right, s_t2, s_t2], axis=0)
        x = x.astype(np.float32)

        ret_dict = {}
        ret_dict['img'] = torch.from_numpy(x)

        axial_sids = des_to_sid['Axial T2']
        axial_volumes = []
        for sid in axial_sids:
            volume, sid = self.get_3d_volume(study_id,
                                             [sid],
                                             'Axial T2')
            axial_volumes.append(volume)

        tmp_imgs = []
        for axial_t2, sid in zip(axial_volumes, axial_sids):
            pred_xy = self.dict_axial_group_idxs[study_id][sid]['pred_xy'].reshape(2, 5, 2)
            pred_z = self.dict_axial_group_idxs[study_id][sid]['pred_z'].reshape(2, 5, 1)
            keypoints = np.concatenate([pred_xy, pred_z], axis=2)

            left_keypoints = keypoints[0]
            right_keypoints = keypoints[1]

            # print(keypoints)
            axial_t2_left, _ = self.crop_out_by_keypoints(axial_t2, left_keypoints,
                                                          z_imgs=3,
                                                          crop_size=160,
                                                          transform=self.axial_transform
                                                          )
            axial_t2_right, _ = self.crop_out_by_keypoints(axial_t2, right_keypoints,
                                                           z_imgs=3,
                                                           crop_size=160,
                                                           transform=self.axial_transform
                                                           )
            # 10, 3, 128, 128
            axial_imgs = np.concatenate([axial_t2_left, axial_t2_right], axis=0)
            tmp_imgs.append(axial_imgs)
        tmp_imgs = np.concatenate(tmp_imgs, axis=0)  # 10*num_series,3,h,w
        tmp_imgs = tmp_imgs.astype(np.float32)
        ret_dict['axial_imgs'] = torch.from_numpy(tmp_imgs)
        ret_dict['study_id'] = str(study_id)
        cond = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
        cond = np.array(cond, dtype=np.int64)
        ret_dict['cond'] = torch.from_numpy(cond)

        return ret_dict


def collate_fn(batch):
    img = []
    axial_imgs = []
    study_id = []
    cond = []

    max_n = 0
    n_list = []
    for item in batch:
        img.append(item['img'])
        a = item['axial_imgs']
        n, _, h, w = a.shape
        n_list.append(n)
        if n > max_n:
            max_n = n
        axial_imgs.append(a)
        study_id.append(item['study_id'])
        cond.append(item['cond'])

    if max_n != 10:
        bs = len(axial_imgs)
        padded_axial_imgs = torch.zeros((bs, max_n, 3, 160, 160), dtype=torch.float32)
        for i in range(bs):
            padded_axial_imgs[i, :n_list[i]] = axial_imgs[i]
        axial_imgs = padded_axial_imgs
    else:
        axial_imgs = torch.stack(axial_imgs, dim=0)

    return {'img': torch.stack(img, dim=0),
            'axial_imgs': axial_imgs,
            'study_id': study_id,
            'cond': torch.stack(cond, dim=0),
            'n_list': torch.LongTensor(n_list),
            }


# def collate_fn(batch):
#     img = []
#     axial_imgs = []
#     cond = []
#
#     for item in batch:
#         img.append(item['img'])
#         axial_imgs.append(item['axial_imgs'])
#         cond.append(item['cond'])
#
#     return {'img': torch.stack(img, dim=0),
#             'axial_imgs': axial_imgs,  # torch.stack(axial_imgs, dim=0),#axial_imgs,
#             'cond': torch.stack(cond, dim=0),
#             }


###
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
    new_df = pd.DataFrame()
    data_root = '/home/hw/ssd_01_new/kaggle/rsna-2024-lumbar-spine-degenerative-classification/'
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


##

if __name__ == '__main__':

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    MODEL_NAME = 'densenet161'
    IN_CHANS = 5
    N_CLASSES = 75
    N_LABELS = 25
    # OUTPUT_DIR = 'exp_v5_5fold_all_view/convnext_small.in12k_ft_in1k_384_lr_0.0006/'

    OUTPUT_DIR = 'wkdir/export/0819_en2_3966/'



    axial_model_fns = [
        'axial_3d_keypoint_model_dense161.pt',
    ]
    axial_keypoint_models = []
    for fn in axial_model_fns:
        fn = f'{OUTPUT_DIR}/keypoint/{fn}'
        print('load: ', fn)
        m = RSNA24Model_Keypoint_3D_Axial(model_name='densenet161').cuda()
        m.load_state_dict(torch.load(fn))
        m.eval()
        m = nn.DataParallel(m)
        axial_keypoint_models.append(m)

    SEED = 8620
    setup_seed(SEED, deterministic=True)

    skf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)

    #### val
    cv = 0

    y_preds_ema = []
    labels = []
    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion2 = nn.CrossEntropyLoss(weight=weights)
    scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        # if fold > 0:
        #     break
        print('#' * 30)
        print(f'start fold{fold}')
        print('#' * 30)
        # df_valid = df.iloc[val_idx]

        folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
        valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index
        folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)
        df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)

        study_ids = list(df_valid['study_id'].unique())

        ## infer the axial 3d keypoint
        dset = RSNA24DatasetTest_LHW_keypoint_3D_Axial(data_root,
                                                       'train_series_descriptions.csv',
                                                       study_ids,
                                                       image_dir=data_root + '/train_images/')

        dloader = DataLoader(dset, batch_size=8, num_workers=8)

        study_id_to_pred_keypoints_axial = {}
        for volumns, study_ids_, sids, depths in tqdm.tqdm(dloader):
            bs, _, _, _ = volumns.shape
            with torch.no_grad():
                with autocast:
                    keypoints = []
                    for i in range(len(axial_keypoint_models)):
                        p = axial_keypoint_models[i](volumns.cuda()).unsqueeze(0)
                        keypoints.append(p)
                    keypoints = torch.cat(keypoints,dim=0).mean(0)
                    keypoints = keypoints.cpu().numpy().reshape(bs, 10, 3)
            for idx, study_id in enumerate(study_ids_):
                study_id = int(study_id)
                sid = int(sids[idx])
                d = depths[idx]
                if study_id not in study_id_to_pred_keypoints_axial.keys():
                    study_id_to_pred_keypoints_axial[study_id] = {}
                study_id_to_pred_keypoints_axial[study_id][sid] = {
                    'points': keypoints[idx],
                    'd': int(d)
                }

        saggital_model_fns = [
            f'best_fold_{fold}_ema.pt',
            # 'best_fold_1_ema.pt',
            # 'best_fold_2_ema.pt'
        ]
        sag_keypoint_models = []
        for fn in saggital_model_fns:
            fn = f'{OUTPUT_DIR}/keypoint/sag/{fn}'
            print('load: ', fn)
            m = RSNA24Model_Keypoint_3D_Sag(model_name='densenet161').cuda()
            m.load_state_dict(torch.load(fn))
            m.eval()
            m = nn.DataParallel(m)
            sag_keypoint_models.append(m)

        ## infer the sag 3d keypoint
        dset = RSNA24DatasetTest_LHW_keypoint_3D_Saggital(data_root,
                                                       'train_series_descriptions.csv',
                                                       study_ids,
                                                       image_dir=data_root + '/train_images/')

        dloader = DataLoader(dset, batch_size=8, num_workers=8)

        study_id_to_pred_keypoints_sag = {}
        for volumns, study_ids_ in tqdm.tqdm(dloader):
            bs,_, _, _, _ = volumns.shape
            with torch.no_grad():
                with autocast:
                    keypoints = []
                    for i in range(len(sag_keypoint_models)):
                        p = sag_keypoint_models[i](volumns.cuda()).unsqueeze(0)
                        keypoints.append(p)
                    keypoints = torch.cat(keypoints, dim=0).mean(0)
                    keypoints = keypoints.cpu().numpy().reshape(bs, 15, 3)
            for idx, study_id in enumerate(study_ids_):
                study_id = int(study_id)
                study_id_to_pred_keypoints_sag[study_id] = {
                    'points': keypoints[idx],
                }

        study_ids = list(df_valid['study_id'].unique())
        valid_ds = RSNA24DatasetTest_LHW_V2_2(data_root,
                                              'train_series_descriptions.csv',
                                              study_ids,
                                              study_id_to_pred_keypoints_sag,
                                              study_id_to_pred_keypoints_axial,
                                              image_dir=data_root + 'train_images/')

        valid_dl = DataLoader(
            valid_ds,
            batch_size=8,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=12,
            collate_fn=collate_fn
        )

        modele = Ensemble_Model(pretrained=False).cuda()
        fname = f'{OUTPUT_DIR}/model_fold{fold}.pth'
        modele.load_state_dict(torch.load(fname))
        modele.eval()
        modele = nn.DataParallel(modele)

        fold_preds = []
        with tqdm.tqdm(valid_dl, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_d in enumerate(pbar):

                    x = tensor_d['img']
                    axial_imgs = tensor_d['axial_imgs']
                    n_list = tensor_d['n_list']

                    x = x.cuda()
                    axial_imgs = axial_imgs.cuda()
                    # for j in range(len(axial_imgs)):
                    #     axial_imgs[j] = axial_imgs[j].cuda()

                    cond = tensor_d['cond'].cuda()

                    with autocast:
                        ye = modele(x, axial_imgs, cond,n_list)
                        for col in range(N_LABELS):
                            pred = ye[:, col * 3:col * 3 + 3]
                            y_preds_ema.append(pred.float().cpu())

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
        scores.append(c)

print(scores)
print('mean score: ', np.mean(scores))
