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

from src.config import G_IMG_SIZE
from src.utils.aux_info import get_train_study_aux_info

# normal, modrate, severe
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

# Axial (x, y, z)   (5)
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


def build_axial_group_idxs(data_root, img_size=512, ext=1.0):
    # pred_keypoints_infos0 = pickle.load(
    #     open(f'{data_root}/v2_axial_3d_keypoints_model0.pkl', 'rb'))
    # pred_keypoints_infos1 = pickle.load(
    #     open(f'{data_root}/v2_axial_3d_keypoints_model1.pkl', 'rb'))
    # pred_keypoints_infos0 = pickle.load(
    #     open(f'{data_root}/pred_keypoints/v2_axial_3d_keypoints_en.pkl', 'rb'))
    pred_keypoints_infos0 = pickle.load(
        open(f'{data_root}/pred_keypoints/v2_axial_3d_keypoints_en3_folds8.pkl', 'rb'))

    # pred_keypoints_infos0 = pickle.load(
    #     open(f'{data_root}/pred_keypoints/v2_axial_3d_keypoints_f0_folds8.pkl', 'rb'))

    dict_axial_group_idxs = {}
    for study_id in pred_keypoints_infos0.keys():
        for sid in pred_keypoints_infos0[study_id].keys():
            pred_info0 = pred_keypoints_infos0[study_id][sid]
            #pred_info1 = pred_keypoints_infos1[study_id][sid]
            predict_keypoints = pred_info0['points']
            #predict_keypoints = (pred_info0['points'] + pred_info1['points']) /2
            depth = int(pred_info0['d'])

            scale = img_size / 4.0
            scale_z = depth / 16.0
            predict_keypoints[:, :2] = predict_keypoints[:, :2] * scale
            predict_keypoints[:, 2] = predict_keypoints[:, 2] * scale_z

            pred_z = predict_keypoints[:, 2].reshape(2, 5)
            pred_z = np.round(pred_z)
            group_idxs = gen_level_group_idxs(pred_z.mean(axis=0), depth, ext)
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

class RSNA24Dataset_Cls_V2(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 transforms_axial=None,
                 in_channels=5,
                 level_by_level = False,
                 axial_crop_xy_size = -1,
                 axial_margin_extend = 1.0,
                 sag_img_size = 128,
                 ):

        self.df = df
        self.aux_info = aux_info
        self.transform = transform
        self.transforms_axial = transforms_axial

        self.phase = phase
        self.in_channels = in_channels
        self.volume_data_root = f'{data_root}/train_images_preprocessed/'

        self.coord_label_df = pd.read_csv(f"{data_root}/train_label_coordinates.csv")
        level2int = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        self.coord_label_df = self.coord_label_df.replace(level2int)

        # keypoints_infos = pickle.load(
        #     open(f'{data_root}/study_id_to_pred_keypoints.pkl', 'rb'))
        keypoints_infos = pickle.load(
            open(f'{data_root}/pred_keypoints/v20_sag_2d_keypoints_center_slice.pkl', 'rb'))

        # keypoints_infos = pickle.load(
        #     open(f'{data_root}/pred_keypoints/v20_sag_2d_keypoints_center_slice_f0.pkl', 'rb'))

        self.keypoints_infos = {}
        for k, v in keypoints_infos.items():
            self.keypoints_infos[int(k)] = v

        self.dict_axial_group_idxs = build_axial_group_idxs(data_root,
                                                            img_size=G_IMG_SIZE,
                                                            ext=axial_margin_extend)

        self.img_size = sag_img_size

        # self.sag_total_slices = 16#10
        # # for example s_t2 keep only the center 5 imgs
        # # Spinal Canal Stenosis
        # self.sag_t2_index_range = [4, 12]#[3, 8]
        # # Left Neural Foraminal Narrowing
        # self.sag_t1_left_index_range = [8, 16]#[5, 10]
        # # Right Neural Foraminal Narrowing
        # self.sag_t1_right_index_range = [0, 8]#[0, 5]
        # self.random_shift_sag_keypoint_prob = 0.2
        # self.axial_imgs_one_level = 7

        self.sag_total_slices = 10
        self.sag_t2_index_range = [3, 8]
        self.sag_t1_left_index_range = [5, 10]
        self.sag_t1_right_index_range = [0, 5]
        self.random_shift_sag_keypoint_prob = 0.2
        self.axial_imgs_one_level = 6

        self.axial_with_xy_crop = True if axial_crop_xy_size!=-1 else False
        self.axial_crop_xy_size = axial_crop_xy_size

        self.level_by_level = level_by_level
        if level_by_level:
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
            print('samples: ', len(self.samples))

    def __len__(self):
        if self.level_by_level:
            return len(self.samples)
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
        # step = length / 100
        # st = length / 2.0 - 4.0 * step
        # end = length + 0.0001
        step = length / self.sag_total_slices
        s = (self.sag_total_slices -2)/ 2.0
        st = length / 2.0 - s * step
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

    def crop_out_by_keypoints(self, volume, keypoints):
        sub_volume_list = []
        att_mask_list = []
        for p in keypoints:
            x, y = int(p[0]), int(p[1])
            bbox = [x - self.img_size // 2,
                    y - self.img_size // 2,
                    x + self.img_size // 2,
                    y + self.img_size // 2]
            # bbox = np.clip(bbox, 0, 512)
            # bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

            # 如果bbox在边界，偏移它以使其具有正确的img_size大小
            if bbox[0] < 0:
                bbox[2] = self.img_size
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[3] = self.img_size
                bbox[1] = 0
            if bbox[2] > G_IMG_SIZE:
                bbox[0] = G_IMG_SIZE - self.img_size
                bbox[2] = G_IMG_SIZE
            if bbox[3] > G_IMG_SIZE:
                bbox[1] = G_IMG_SIZE - self.img_size
                bbox[3] = G_IMG_SIZE

            bbox = [int(e) for e in bbox]
            v = volume[:, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
            # print(v.shape)
            # att_mask = np.zeros((1, self.img_size, self.img_size), dtype=np.uint8)
            # x = int(x - bbox[0])
            # y = int(y - bbox[1])
            # att_mask[0] = cv2.circle(att_mask[0], (x, y), 5, (255, 255, 255), 9)
            # att_mask_list.append(att_mask)

            sub_volume_list.append(v)

        # att_mask = np.array(att_mask_list)
        # att_mask = np.asarray(att_mask, dtype=np.float32) / 255.0
        volume_crop = np.array(sub_volume_list)
        return volume_crop, None  # att_mask

    def get_axial_coord(self, study_id, series_id):
        coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
        coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
        # print('coord_sub_df: ', len(coord_sub_df))
        keypoints = -np.ones((5, 2, 3), np.float32)  # 5_level, left/right, xyz

        for _, row in coord_sub_df.iterrows():
            idx = 0
            if row['condition'] == 'Right Subarticular Stenosis':
                idx = 1
            x, y, z = row['x'], row['y'], row['instance_number']

            keypoints[row['level'], idx, 0] = x
            keypoints[row['level'], idx, 1] = y
            keypoints[row['level'], idx, 2] = z

        #keypoints = keypoints.reshape(-1, 3)
        return keypoints

    def __getitem__(self, idx):
        if self.level_by_level:
            item = self.samples[idx]
            study_id = int(item['study_id'])
            label = item['target']
            level = item['level']

            keypoints = self.keypoints_infos[study_id]
            s_t1_keypoints = keypoints[0][level: level + 1]
            s_t2_keypoints = keypoints[1][level: level + 1]
            levels = [level]
        else:
            item = self.df.iloc[idx]
            study_id = int(item['study_id'])
            label = item[1:].values.astype(np.int64)
            # label = label[:15]  # without Axial (Stenosis)
            keypoints = self.keypoints_infos[study_id]
            s_t1_keypoints = keypoints[0]
            s_t2_keypoints = keypoints[1]
            levels = range(5)

        # use CS  Saggital T2 only
        #s_t1_keypoints = s_t2_keypoints
        if self.phase=='train' and random.random() < self.random_shift_sag_keypoint_prob:
            noise1 = np.random.uniform(-3.0, 3.0, s_t1_keypoints.shape)
            s_t1_keypoints += noise1
            noise2 = np.random.uniform(-3.0, 3.0, s_t2_keypoints.shape)
            s_t2_keypoints += noise2


        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        if 'Sagittal T1' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((self.sag_total_slices, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            # TODO optimisze this
            label[5:15] = 0
            s_t1_keypoints = np.zeros((5, 2), dtype=np.int64)
            s_t1_keypoints[:] = G_IMG_SIZE // 2

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T1'])
            index = self.gen_slice_index_Sagittal(length=len(fns))
            s_t1 = self.load_volume(fns, index, bbox=None)

        # Sagittal T2/STIR
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((self.sag_total_slices, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            label[:5] = 0
            s_t2_keypoints = np.zeros((5, 2), dtype=np.int64)
            s_t2_keypoints[:] = G_IMG_SIZE // 2

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T2/STIR'])
            index = self.gen_slice_index_Sagittal(length=len(fns))
            s_t2 = self.load_volume(fns, index, bbox=None)

        if self.phase == 'train':
            s_t1 = self.drop_img(s_t1, prob=0.1)
            s_t2 = self.drop_img(s_t2, prob=0.1)

        r = self.sag_t2_index_range
        s_t2 = s_t2[r[0]: r[1]]

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)

            augmented1 = self.transform(image=s_t1, keypoints=s_t1_keypoints)
            s_t1 = augmented1['image']
            s_t1_keypoints = augmented1['keypoints']

            augmented2 = self.transform(image=s_t2, keypoints=s_t2_keypoints)
            s_t2 = augmented2['image']
            s_t2_keypoints = augmented2['keypoints']

            s_t1 = s_t1.transpose(2, 0, 1)
            s_t2 = s_t2.transpose(2, 0, 1)

        # crop out by keypoints
        # s_t2 (512, 512, 10)
        # (5, 2)
        # (5, 128, 128, 5)
        s_t2, s_t2_mask = self.crop_out_by_keypoints(s_t2, s_t2_keypoints)

        #
        r = self.sag_t1_left_index_range
        s_t1_left = s_t1[r[0]:r[1], :, :]
        #s_t1_left = s_t1[5:, :, :]
        r = self.sag_t1_right_index_range
        #s_t1_right = s_t1[:5, :, :]
        s_t1_right = s_t1[r[0]:r[1], :, :]

        s_t1_left, mask = self.crop_out_by_keypoints(s_t1_left, s_t1_keypoints)
        # s_t1_left = np.concatenate([s_t1_left, s_t2], axis=1)
        s_t1_right, mask = self.crop_out_by_keypoints(s_t1_right, s_t1_keypoints)
        # s_t1_right = np.concatenate([s_t1_right, s_t2], axis=1)

        # s_t1_5 = s_t1[3:8, :, :]
        # s_t1_5, _ = self.crop_out_by_keypoints(s_t1_5, s_t1_keypoints)
        # s_t2 = np.concatenate([s_t2, s_t1_5], axis=1)

        # Axial
        fns, sid = self.get_fns(study_id, des_to_sid['Axial T2'])
        axial_imgs = []

        axial_gt_pts = self.get_axial_coord(study_id, sid)
        if self.level_by_level:
            p = axial_gt_pts[levels[0]]
            # left
            if p[0][2] < 0:
                label[3] = -100
            # right
            if p[1][2] < 0:
                label[4] = -100
        else:
            for level in levels:
                p = axial_gt_pts[level]
                # left
                if p[0][2] < 0:
                    label[3, level] = -100
                # right
                if p[1][2] < 0:
                    label[4, label] = -100

        for level in levels:
            r = self.dict_axial_group_idxs[study_id][sid]['group_idxs'][level]
            indexes = list(range(r[0], r[1] + 1))
            imgs = self.load_volume(fns, indexes)
            imgs = imgs.transpose(1, 2, 0)
            # 6, 256, 256
            if self.axial_with_xy_crop:
                center_xy = self.dict_axial_group_idxs[study_id][sid]['center_xy'][level]
                bbox = gen_bbox_by_center_xy(center_xy,
                                             crop_size=self.axial_crop_xy_size,
                                             img_size=G_IMG_SIZE)
                imgs = imgs[bbox[1]: bbox[3], bbox[0]: bbox[2], :].copy()

            ind = select_elements(imgs.shape[-1], self.axial_imgs_one_level,
                                  randomize=True if self.phase == 'train' else False)
            imgs_ = imgs[:, :, ind]
            ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
            ind_a = np.where(np.array(ind) + 1 >= imgs.shape[-1] - 1, imgs.shape[-1] - 1, np.array(ind) + 1)
            imgs = np.stack([imgs[:, :, ind_b], imgs_, imgs[:, :, ind_a]], axis=2)

            imgs = np.stack([self.transforms_axial(image=imgs[:, :, :, i])['image'] for i in range(imgs.shape[-1])])
            # 6, 3, h, w
            #imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2))).float()
            imgs = imgs.transpose((0, 3, 1, 2))
            axial_imgs.append(imgs)

        # 5, 6, 3, h, w
        axial_imgs = np.array(axial_imgs)
        axial_imgs = axial_imgs.astype(np.float32)
        if self.level_by_level:
            # 6, 3, h, w
            axial_imgs = axial_imgs[0]

        ## !! must first s_t2, corresponding to label
        x = np.concatenate([s_t2, s_t1_left, s_t1_right, s_t2, s_t2], axis=0)
        x = x.astype(np.float32)
        if self.level_by_level:
            cond = [0] * 1 + [1] * 1 + [2] * 1 + [3] * 1 + [4] * 1
        else:
            cond = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
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


class RSNA24Dataset_Cls_V2_Level_by_Level(Dataset):
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
        keypoints_infos = pickle.load(
            open(f'{data_root}/study_id_to_pred_keypoints.pkl', 'rb'))
        self.keypoints_infos = {}
        for k, v in keypoints_infos.items():
            self.keypoints_infos[int(k)] = v

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

        print('[RSNA24Dataset_Cls_V2_Level_by_Level] samples: ', len(self.samples))

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
        att_mask_list = []
        for p in keypoints:
            x, y = int(p[0]), int(p[1])
            bbox = [x - self.img_size // 2,
                    y - self.img_size // 2,
                    x + self.img_size // 2,
                    y + self.img_size // 2]
            # bbox = np.clip(bbox, 0, 512)
            # bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

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
            v = volume[:, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
            # print(v.shape)
            # att_mask = np.zeros((1, self.img_size, self.img_size), dtype=np.uint8)
            # x = int(x - bbox[0])
            # y = int(y - bbox[1])
            # att_mask[0] = cv2.circle(att_mask[0], (x, y), 5, (255, 255, 255), 9)
            # att_mask_list.append(att_mask)

            sub_volume_list.append(v)

        # att_mask = np.array(att_mask_list)
        # att_mask = np.asarray(att_mask, dtype=np.float32) / 255.0
        volume_crop = np.array(sub_volume_list)
        return volume_crop, None  # att_mask

    def __getitem__(self, idx):
        item = self.samples[idx]
        study_id = int(item['study_id'])
        label = item['target']
        level = item['level']

        keypoints = self.keypoints_infos[study_id]
        s_t1_keypoints = keypoints[0][level: level + 1] # 1, 5, 2
        s_t2_keypoints = keypoints[1][level: level + 1]

        # print('s_t1_keypoints shape: ', s_t1_keypoints.shape)
        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        if 'Sagittal T1' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, 512, 512), dtype=np.uint8)
            # TODO optimisze this
            label[5:15] = 0
            s_t1_keypoints = np.zeros((1, 2), dtype=np.int64)
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
            s_t2_keypoints = np.zeros((1, 2), dtype=np.int64)
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

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)

            augmented1 = self.transform(image=s_t1, keypoints=s_t1_keypoints)
            s_t1 = augmented1['image']
            s_t1_keypoints = augmented1['keypoints']

            augmented2 = self.transform(image=s_t2, keypoints=s_t2_keypoints)
            s_t2 = augmented2['image']
            s_t2_keypoints = augmented2['keypoints']

            s_t1 = s_t1.transpose(2, 0, 1)
            s_t2 = s_t2.transpose(2, 0, 1)

        # crop out by keypoints
        try:
            s_t2, s_t2_mask = self.crop_out_by_keypoints(s_t2, s_t2_keypoints)
            # s_t2 = np.concatenate([s_t2, s_t2_mask], axis=1)

            #
            s_t1_left = s_t1[5:, :, :]
            s_t1_right = s_t1[:5, :, :]

            s_t1_left, mask = self.crop_out_by_keypoints(s_t1_left, s_t1_keypoints)
            # s_t1_left = np.concatenate([s_t1_left, mask], axis=1)
            # s_t1_left = np.concatenate([s_t1_left, s_t2], axis=1)
            s_t1_right, mask = self.crop_out_by_keypoints(s_t1_right, s_t1_keypoints)
            # s_t1_right = np.concatenate([s_t1_right, mask], axis=1)
            # s_t1_right = np.concatenate([s_t1_right, s_t2], axis=1)

            # s_t1_5 = s_t1[3:8, :, :]
            # s_t1_5, _ = self.crop_out_by_keypoints(s_t1_5, s_t1_keypoints)
            # s_t2 = np.concatenate([s_t2, s_t1_5], axis=1)

        except Exception as e:
            print(study_id)
            print(s_t2_keypoints)
            print(s_t1_keypoints)
            print(e)
            exit(0)

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
            's_t2': s_t2,
            's_t1_left': s_t1_left,
            's_t1_right': s_t1_right,
            'label': label,
            'cond': cond
        }
