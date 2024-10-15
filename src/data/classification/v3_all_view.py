# -*- coding: utf-8 -*-
import json
from glob import glob

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import pickle
import random

from notebooks.infer_cascade2 import load_dicom_line_par
from src.utils.aux_info import get_train_study_aux_info

cond_to_idx = {
    'Spinal Canal Stenosis': 0,
    'Left Neural Foraminal Narrowing': 1,
    'Right Neural Foraminal Narrowing': 2,
    'Left Subarticular Stenosis': 3,
    'Right Subarticular Stenosis': 4
}


def select_elements(K, N, randomize=True):
    lst = list(range(K))
    length = K

    if length <= N:
        # K+1 <= N の場合
        repeat_times = (N // length) + 1
        lst = sorted(lst * repeat_times)

    if randomize and len(lst) >= N:
        result = []
        interval = len(lst) / N
        for i in range(N):
            index = int((i + random.choice([-0.3, 0, 0.3])) * interval)
            # print(index)
            index = max(0, min(len(lst) - 1, index))
            result.append(lst[index])
        result = sorted(result)
    else:
        interval = len(lst) / N
        result = [lst[int(i * interval)] for i in range(N)]
    return result


def gen_level_group_idxs(pred_z_list, depth):
    z1, z2, z3, z4, z5 = pred_z_list
    margin1 = abs(z2 - z1) / 2
    z1_start = z1 - margin1
    z1_end = z1 + margin1

    margin2 = abs(z3 - z2) / 2
    z2_start = z2 - margin1
    z2_end = z2 + margin2

    margin3 = abs(z4 - z3) / 2
    z3_start = z3 - margin2
    z3_end = z3 + margin3

    margin4 = abs(z5 - z4) / 2
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


def build_axial_group_idxs(data_root, img_size=512):
    pred_keypoints_infos0 = pickle.load(
        open(f'{data_root}/v2_axial_3d_keypoints_model0.pkl', 'rb'))
    pred_keypoints_infos1 = pickle.load(
        open(f'{data_root}/v2_axial_3d_keypoints_model1.pkl', 'rb'))
    dict_axial_group_idxs = {}
    for study_id in pred_keypoints_infos0.keys():
        for sid in pred_keypoints_infos0[study_id].keys():
            pred_info0 = pred_keypoints_infos0[study_id][sid]
            pred_info1 = pred_keypoints_infos1[study_id][sid]
            predict_keypoints = pred_info0['points']
            #predict_keypoints = (pred_info0['points'] + pred_info1['points']) /2
            depth = int(pred_info0['d'])

            scale = img_size / 4.0
            scale_z = depth / 16.0
            predict_keypoints[:, :2] = predict_keypoints[:, :2] * scale
            predict_keypoints[:, 2] = predict_keypoints[:, 2] * scale_z

            pred_z = predict_keypoints[:, 2].reshape(2, 5)
            pred_z = np.round(pred_z.mean(axis=0))
            group_idxs = gen_level_group_idxs(pred_z, depth)
            group_idxs = np.asarray(group_idxs, dtype=np.int64)

            xy = predict_keypoints[:, :2].reshape(2, 5, 2)
            center_xy = np.round(xy.mean(axis=0))

            if study_id not in dict_axial_group_idxs.keys():
                dict_axial_group_idxs[study_id] = {}
            dict_axial_group_idxs[study_id][sid] = {
                'group_idxs': group_idxs,
                'center_xy': center_xy
            }
    return dict_axial_group_idxs

def gen_bbox_by_center_xy(center_xy, crop_size=256, img_size=512):
    x, y = int(center_xy[0]), int(center_xy[1])
    bbox = [x - crop_size // 2,
            y - crop_size // 2,
            x + crop_size // 2,
            y + crop_size // 2]

    if bbox[0] < 0:
        bbox[2] = crop_size
        bbox[0] = 0
    if bbox[1] < 0:
        bbox[3] = crop_size
        bbox[1] = 0
    if bbox[2] > img_size:
        bbox[0] = img_size - crop_size
        bbox[2] = img_size
    if bbox[3] > img_size:
        bbox[1] = img_size - crop_size
        bbox[3] = img_size
    return bbox

class RSNA24Dataset_Cls_V3(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 transforms_axial=None,
                 in_channels=5):

        self.df = df
        self.aux_info = aux_info
        self.transform = transform
        self.transforms_axial = transforms_axial

        self.phase = phase
        self.in_channels = in_channels
        self.volume_data_root = f'{data_root}/train_images_preprocessed/'

        keypoints_infos0 = pickle.load(
            open(f'{data_root}/v2_sag_2d_keypoints_model0.pkl', 'rb'))
        self.keypoints_infos0 = {}
        for k, v in keypoints_infos0.items():
            self.keypoints_infos0[int(k)] = v

        keypoints_infos1 = pickle.load(
            open(f'{data_root}/v2_sag_2d_keypoints_model1.pkl', 'rb'))
        self.keypoints_infos1 = {}
        for k, v in keypoints_infos1.items():
            self.keypoints_infos1[int(k)] = v

        self.dict_axial_group_idxs = build_axial_group_idxs(data_root)
        self.img_size = 128

    def __len__(self):
        return len(self.df)

    def load_volume(self, fns, index, bbox=None):
        arr = []
        for i in index:
            a = cv2.imread(fns[i], 0)
            if bbox is not None:
                a = a[bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
                try:
                    a = cv2.resize(a, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    print(e)
                    print(bbox)
                    print(fns)
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

    def drop_img(self, volume, prob=0.1):
        for i in range(len(volume)):
            # keep the center for avoiding dropped all
            if i == len(volume) // 2:
                continue
            if np.random.random() < prob:
                volume[i] *= 0
        return volume

    def get_fns(self, study_id, sids):
        if self.phase != 'train':
            # TODO use multi when test
            id = sids[0]
            # max_id = -1
            # max_fns = []
            # max_n = 0
            # for id in sids:
            #     fns = glob(f'{self.volume_data_root}/{study_id}/{id}/*.png')
            #     fns = sorted(fns)
            #     if len(fns) > max_n:
            #         max_n = len(fns)
            #         max_fns = fns
            #         max_id = id
            # return max_fns, max_id
        else:
            id = random.choice(sids)
        fns = glob(f'{self.volume_data_root}/{study_id}/{id}/*.png')
        fns = sorted(fns)
        return fns, id

    def crop_out_by_keypoints(self, volume, keypoints_all):
        # keypoints 5_depth, 5_level, 2
        d, levels, _ = keypoints_all.shape
        assert volume.shape[0] == d

        sub_volume_list = []
        for le in range(levels):
            keypoints = keypoints_all[:, le, :]
            for ii, p in enumerate(keypoints):
                x, y = int(p[0]), int(p[1])
                bbox = [x - self.img_size // 2,
                        y - self.img_size // 2,
                        x + self.img_size // 2,
                        y + self.img_size // 2]

                # 如果bbox在边界，偏移它以使其具有正确的img_size大小
                if bbox[0] < 0:
                    bbox[2] = self.img_size
                    bbox[0] = 0
                if bbox[1] < 0:
                    bbox[3] = self.img_size
                    bbox[1] = 0
                if bbox[2] > 512:
                    bbox[0] = 512 - self.img_size
                    bbox[2] = 512
                if bbox[3] > 512:
                    bbox[1] = 512 - self.img_size
                    bbox[3] = 512

                bbox = [int(e) for e in bbox]
                v = volume[ii, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()

                sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        #print(volume_crop.shape)
        _, h, w = volume_crop.shape
        # 5_level, 5_depth, h, w
        volume_crop = volume_crop.reshape(levels, 5, h, w)
        return volume_crop, None  # att_mask



    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        label = item[1:].values.astype(np.int64)
        # label = label[:15]  # without Axial (Stenosis)
        # keypoints = self.keypoints_infos[study_id]
        # s_t1_keypoints = keypoints[0]
        # s_t2_keypoints = keypoints[1]

        keypoints = self.keypoints_infos0[study_id]  # (2, 10, 5, 2)
        s_t1_keypoints = keypoints[0]  # 10, 5, 2
        s_t2_keypoints = keypoints[1]  # 10, 5, 2

        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        if 'Sagittal T1' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, 512, 512), dtype=np.uint8)
            # TODO optimisze this
            label[5:15] = 0
            s_t1_keypoints = np.zeros((10, 5, 2), dtype=np.int64)
            s_t1_keypoints[:] = 256
            # s_t1_space = -1

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T1'])
            index = self.gen_slice_index_Sagittal(length=len(fns))
            s_t1 = self.load_volume(fns, index, bbox=None)
            # s_t1_space = self.get_spacing_for_crop(study_id, sid)

        # Sagittal T2/STIR
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10, 512, 512), dtype=np.uint8)
            label[:5] = 0
            s_t2_keypoints = np.zeros((10, 5, 2), dtype=np.int64)
            s_t2_keypoints[:] = 256
            # s_t2_space = -1

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T2/STIR'])
            index = self.gen_slice_index_Sagittal(length=len(fns))
            s_t2 = self.load_volume(fns, index, bbox=None)
            # s_t2_space = self.get_spacing_for_crop(study_id, sid)

        if self.phase == 'train':
            s_t1 = self.drop_img(s_t1, prob=0.1)
            s_t2 = self.drop_img(s_t2, prob=0.1)

        # s_t2 keep only the center 5 imgs
        s_t2 = s_t2[3: 8]
        s_t2_keypoints = s_t2_keypoints[3: 8]

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)
            s_t1_keypoints = s_t1_keypoints.reshape(50, 2)
            augmented1 = self.transform(image=s_t1, keypoints=s_t1_keypoints)
            s_t1 = augmented1['image']
            s_t1_keypoints = augmented1['keypoints']
            s_t1_keypoints = np.array(s_t1_keypoints)
            s_t1_keypoints = s_t1_keypoints.reshape(10, 5, 2)

            s_t2_keypoints = s_t2_keypoints.reshape(25, 2)
            augmented2 = self.transform(image=s_t2, keypoints=s_t2_keypoints)
            s_t2 = augmented2['image']
            s_t2_keypoints = augmented2['keypoints']
            s_t2_keypoints = np.array(s_t2_keypoints)
            s_t2_keypoints = s_t2_keypoints.reshape(5, 5, 2)

            s_t1 = s_t1.transpose(2, 0, 1)
            s_t2 = s_t2.transpose(2, 0, 1)

        # crop out by keypoints
        s_t2, s_t2_mask = self.crop_out_by_keypoints(s_t2, s_t2_keypoints)
        # s_t2 = np.concatenate([s_t2, s_t2_mask], axis=1)

        #
        s_t1_left = s_t1[5:, :, :]
        s_t1_keypoints_left = s_t1_keypoints[5:, ]
        s_t1_right = s_t1[:5, :, :]
        s_t1_keypoints_right = s_t1_keypoints[:5, ]

        s_t1_left, mask = self.crop_out_by_keypoints(s_t1_left, s_t1_keypoints_left)
        # s_t1_left = np.concatenate([s_t1_left, mask], axis=1)
        # s_t1_left = np.concatenate([s_t1_left, s_t2], axis=1)
        s_t1_right, mask = self.crop_out_by_keypoints(s_t1_right, s_t1_keypoints_right)
        # s_t1_right = np.concatenate([s_t1_right, mask], axis=1)
        # s_t1_right = np.concatenate([s_t1_right, s_t2], axis=1)

        # s_t1_5 = s_t1[3:8, :, :]
        # s_t1_5, _ = self.crop_out_by_keypoints(s_t1_5, s_t1_keypoints)
        # s_t2 = np.concatenate([s_t2, s_t1_5], axis=1)

        # Axial
        fns, sid = self.get_fns(study_id, des_to_sid['Axial T2'])

        # data_dir = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
        # volume = load_dicom_line_par(f'{data_dir}/train_images/', study_id, sid)
        axial_imgs = []

        for level in range(5):
            # imgs = np.load(f"{data_dir}/t2A_each/{study_id}__{sid}__{4 - level}.npy")
            # imgs = volume[4 - level]
            r = self.dict_axial_group_idxs[study_id][sid]['group_idxs'][level]
            indexes = list(range(r[0], r[1] + 1))
            imgs = self.load_volume(fns, indexes)
            imgs = imgs.transpose(1, 2, 0)

            # center_xy = self.dict_axial_group_idxs[study_id][sid]['center_xy'][level]
            # bbox = gen_bbox_by_center_xy(center_xy, crop_size=256, img_size=512)
            # imgs = imgs[bbox[1]: bbox[3], bbox[0]: bbox[2], :].copy()

            ind = select_elements(imgs.shape[-1], 6,
                                  randomize=True if self.phase == 'train' else False)
            imgs_ = imgs[:, :, ind]
            ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
            ind_a = np.where(np.array(ind) + 1 >= imgs.shape[-1] - 1, imgs.shape[-1] - 1, np.array(ind) + 1)
            imgs = np.stack([imgs[:, :, ind_b], imgs_, imgs[:, :, ind_a]], axis=2)

            imgs = np.stack([self.transforms_axial(image=imgs[:, :, :, i])['image'] for i in range(imgs.shape[-1])])
            # 6, 3, h, w
            imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2))).float()
            axial_imgs.append(imgs)
        # 5, 6, 3, h, w
        axial_imgs = np.array(axial_imgs)
        axial_imgs = axial_imgs.astype(np.float32)

        ## !! must first s_t2, corresponding to label
        x = np.concatenate([s_t2, s_t1_left, s_t1_right, s_t2, s_t2], axis=0)
        x = x.astype(np.float32)
        cond = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
        cond = np.array(cond, dtype=np.int64)

        return {
            'img': x,
            'axial_imgs': axial_imgs,
            's_t2': s_t2,
            's_t1_left': s_t1_left,
            's_t1_right': s_t1_right,
            'label': label,
            'cond': cond
        }


class RSNA24Dataset_Cls_V3_Level_by_Level(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 transforms_axial=None,
                 in_channels=5):

        self.df = df
        self.aux_info = aux_info
        self.transform = transform
        self.transforms_axial = transforms_axial

        self.phase = phase
        self.in_channels = in_channels
        self.volume_data_root = f'{data_root}/train_images_preprocessed/'
        keypoints_infos0 = pickle.load(
            open(f'{data_root}/v2_sag_2d_keypoints_model0.pkl', 'rb'))
        self.keypoints_infos0 = {}
        for k, v in keypoints_infos0.items():
            self.keypoints_infos0[int(k)] = v

        keypoints_infos1 = pickle.load(
            open(f'{data_root}/v2_sag_2d_keypoints_model1.pkl', 'rb'))
        self.keypoints_infos1 = {}
        for k, v in keypoints_infos1.items():
            self.keypoints_infos1[int(k)] = v

        self.dict_axial_group_idxs = build_axial_group_idxs(data_root)

        self.samples = []
        for i, row in df.iterrows():
            study_id = row['study_id']
            if study_id in [2492114990, 3008676218, 2780132468, 3637444890]:
                continue
            label = row[1:].values.astype(np.int64)
            label = label.reshape(5, 5)
            for level in range(5):
                self.samples.append({
                    'study_id': study_id,
                    'level': level,
                    'target': label[:, level]
                })

        print('[RSNA24Dataset_Cls_V3_Level_by_Level] samples: ', len(self.samples))

        self.img_size = 128

    def __len__(self):
        return len(self.samples)

    def load_volume(self, fns, index, bbox=None):
        arr = []
        for i in index:
            a = cv2.imread(fns[i], 0)
            if bbox is not None:
                a = a[bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
                try:
                    a = cv2.resize(a, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
                except Exception as e:
                    print(e)
                    print(bbox)
                    print(fns)
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

    def drop_img(self, volume, prob=0.1):
        for i in range(len(volume)):
            # keep the center for avoiding dropped all
            if i == len(volume) // 2:
                continue
            if np.random.random() < prob:
                volume[i] *= 0
        return volume

    def get_fns(self, study_id, sids):
        if self.phase != 'train':
            # TODO use multi when test
            id = sids[0]
        else:
            id = random.choice(sids)
        fns = glob(f'{self.volume_data_root}/{study_id}/{id}/*.png')
        fns = sorted(fns)
        return fns, id

    def crop_out_by_keypoints(self, volume, keypoints):
        sub_volume_list = []
        for ii, p in enumerate(keypoints):
            x, y = int(p[0]), int(p[1])
            bbox = [x - self.img_size // 2,
                    y - self.img_size // 2,
                    x + self.img_size // 2,
                    y + self.img_size // 2]

            # 如果bbox在边界，偏移它以使其具有正确的img_size大小
            if bbox[0] < 0:
                bbox[2] = self.img_size
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[3] = self.img_size
                bbox[1] = 0
            if bbox[2] > 512:
                bbox[0] = 512 - self.img_size
                bbox[2] = 512
            if bbox[3] > 512:
                bbox[1] = 512 - self.img_size
                bbox[3] = 512

            bbox = [int(e) for e in bbox]
            v = volume[ii, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()

            sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        volume_crop = volume_crop[np.newaxis, :]
        return volume_crop, None  # att_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        study_id = int(item['study_id'])
        label = item['target']
        level = item['level']

        keypoints = self.keypoints_infos0[study_id] # (2, 10, 5, 2)
        s_t1_keypoints = keypoints[0][:, level] # 10, 2
        s_t2_keypoints = keypoints[1][:, level] # 10, 2


        # print('s_t1_keypoints shape: ', s_t1_keypoints.shape)
        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        if 'Sagittal T1' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, 512, 512), dtype=np.uint8)
            # TODO optimisze this
            label[5:15] = 0
            s_t1_keypoints = np.zeros((10, 2), dtype=np.int64)
            s_t1_keypoints[:] = 256
            # s_t1_space = -1

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T1'])
            index = self.gen_slice_index_Sagittal(length=len(fns))
            s_t1 = self.load_volume(fns, index, bbox=None)
            # s_t1_space = self.get_spacing_for_crop(study_id, sid)

        # Sagittal T2/STIR
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10, 512, 512), dtype=np.uint8)
            label[:5] = 0
            s_t2_keypoints = np.zeros((10, 2), dtype=np.int64)
            s_t2_keypoints[:] = 256
            # s_t2_space = -1

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T2/STIR'])
            index = self.gen_slice_index_Sagittal(length=len(fns))
            s_t2 = self.load_volume(fns, index, bbox=None)
            # s_t2_space = self.get_spacing_for_crop(study_id, sid)

        if self.phase == 'train':
            s_t1 = self.drop_img(s_t1, prob=0.1)
            s_t2 = self.drop_img(s_t2, prob=0.1)

        # s_t2 keep only the center 5 imgs
        s_t2 = s_t2[3: 8]
        s_t2_keypoints = s_t2_keypoints[3 : 8]

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)

            augmented1 = self.transform(image=s_t1, keypoints=s_t1_keypoints)
            s_t1 = augmented1['image']
            s_t1_keypoints = augmented1['keypoints']
            s_t1_keypoints = np.array(s_t1_keypoints)

            augmented2 = self.transform(image=s_t2, keypoints=s_t2_keypoints)
            s_t2 = augmented2['image']
            s_t2_keypoints = augmented2['keypoints']
            s_t1_keypoints = np.array(s_t1_keypoints)

            s_t1 = s_t1.transpose(2, 0, 1)
            s_t2 = s_t2.transpose(2, 0, 1)

        # crop out by keypoints
        s_t2, s_t2_mask = self.crop_out_by_keypoints(s_t2, s_t2_keypoints)
        # s_t2 = np.concatenate([s_t2, s_t2_mask], axis=1)

        #
        s_t1_left = s_t1[5:, :, :]
        s_t1_keypoints_left = s_t1_keypoints[5:, ]
        s_t1_right = s_t1[:5, :, :]
        s_t1_keypoints_right = s_t1_keypoints[:5, ]

        s_t1_left, mask = self.crop_out_by_keypoints(s_t1_left, s_t1_keypoints_left)
        # s_t1_left = np.concatenate([s_t1_left, mask], axis=1)
        # s_t1_left = np.concatenate([s_t1_left, s_t2], axis=1)
        s_t1_right, mask = self.crop_out_by_keypoints(s_t1_right, s_t1_keypoints_right)
        # s_t1_right = np.concatenate([s_t1_right, mask], axis=1)
        # s_t1_right = np.concatenate([s_t1_right, s_t2], axis=1)

        # s_t1_5 = s_t1[3:8, :, :]
        # s_t1_5, _ = self.crop_out_by_keypoints(s_t1_5, s_t1_keypoints)
        # s_t2 = np.concatenate([s_t2, s_t1_5], axis=1)

        # Axial
        fns, sid = self.get_fns(study_id, des_to_sid['Axial T2'])

        r = self.dict_axial_group_idxs[study_id][sid]['group_idxs'][level]
        indexes = list(range(r[0], r[1] + 1))
        imgs = self.load_volume(fns, indexes)

        imgs = imgs.transpose(1, 2, 0)
        # center_xy = self.dict_axial_group_idxs[study_id][sid]['center_xy'][level]
        # bbox = gen_bbox_by_center_xy(center_xy, crop_size=256, img_size=512)
        # imgs = imgs[bbox[1]: bbox[3], bbox[0]: bbox[2], :].copy()


        ind = select_elements(imgs.shape[-1], 6,
                              randomize=True if self.phase == 'train' else False)
        imgs_ = imgs[:, :, ind]
        ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
        ind_a = np.where(np.array(ind) + 1 >= imgs.shape[-1] - 1, imgs.shape[-1] - 1, np.array(ind) + 1)
        imgs = np.stack([imgs[:, :, ind_b], imgs_, imgs[:, :, ind_a]], axis=2)

        imgs = np.stack([self.transforms_axial(image=imgs[:, :, :, i])['image'] for i in range(imgs.shape[-1])])
        # 6, 3, h, w
        imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2))).float()
        axial_imgs = imgs

        # 6, 3, h, w
        # axial_imgs = axial_imgs.astype(np.float32)

        ## !! must first s_t2, corresponding to label
        x = np.concatenate([s_t2, s_t1_left, s_t1_right, s_t2, s_t2], axis=0)
        x = x.astype(np.float32)
        cond = [0] * 1 + [1] * 1 + [2] * 1 + [3] * 1 + [4] * 1
        cond = np.array(cond, dtype=np.int64)

        return {
            'img': x,
            'axial_imgs': axial_imgs,
            # 's_t2': s_t2,
            # 's_t1_left': s_t1_left,
            # 's_t1_right': s_t1_right,
            'label': label,
            'cond': cond
        }


def convert_to_cv2_img(volume):
    k, d, h, w = volume.shape
    img = np.zeros((k * h, d * w))
    for ik in range(k):
        for id in range(d):
            img[ik * h: h * (ik + 1), id * w: w * (id + 1)] = volume[ik, id]

    img = 255 * (img - img.min()) / (img.max() - img.min())
    return img


if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # df = df[df['study_id'] == 2780132468]

    aux_info = get_train_study_aux_info(data_root)

    # check
    # for k, v in aux_info.items():
    #     des_to_sid = v['des_to_sid']
    #     print(k, des_to_sid)
    #     s = des_to_sid['Sagittal T2/STIR']

    # df = df[df['study_id'] == 4096820034].copy().reset_index(drop=True)
    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    create_dir(debug_dir)
    import albumentations as A

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=AUG_PROB),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transforms_axial = A.Compose([
        A.Resize(512, 512),
        A.CenterCrop(300, 300),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dset = RSNA24Dataset_Cls_V3_Level_by_Level(data_root, aux_info, df, phase='valid',
    #                                            transform=transforms_train,
    #                                            transforms_axial=transforms_axial)

    dset = RSNA24Dataset_Cls_V3(data_root, aux_info, df, phase='valid',
                                               transform=transforms_train,
                                               transforms_axial=transforms_axial)

    print(len(dset))
    for d in dset:
        print('x: ', d['img'].shape)
        s_t2 = d['s_t2']
        s_t1_left = d['s_t1_left']
        s_t1_right = d['s_t1_right']
        axial_imgs = d['axial_imgs']
        print('img: ', d['img'].shape)
        print('s_t2: ', s_t2.shape)
        print('s_t1_left: ', s_t1_left.shape)
        print('s_t1_right: ', s_t1_right.shape)
        print('axial_imgs: ', axial_imgs.shape)

        cv2.imwrite(f'{debug_dir}/V3_s_t2.jpg', convert_to_cv2_img(s_t2))
        cv2.imwrite(f'{debug_dir}/V3_s_t1_left.jpg', convert_to_cv2_img(s_t1_left))
        cv2.imwrite(f'{debug_dir}/V3_s_t1_right.jpg', convert_to_cv2_img(s_t1_right))
        #cv2.imwrite(f'{debug_dir}/V3_axial.jpg', convert_to_cv2_img(axial_imgs))
        break
