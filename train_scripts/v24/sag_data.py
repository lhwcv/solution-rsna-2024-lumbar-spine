# -*- coding: utf-8 -*-
import math
import os
import random
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F
from train_scripts.v20.dicom import load_dicom, rescale_keypoints_by_meta
from math import sin, cos
from src.utils.comm import create_dir
from train_scripts.v20.hard_list import hard_sag_t2_keypoints_study_id_list, hard_sag_t2_z_keypoints_study_id_list, \
    hard_sag_t1_keypoints_study_id_list, hard_sag_t1_z_keypoints_study_id_list

sag_z_hard_list = [
    364930790,
    2662989538,
    2388577668,
    2530679352,
    2530679352,
    757619082,
    1395773918,
    1879696087,
    2135829458,
    3713534743,
    3781188430,
    4201106871,
    1647904243,
    2830065820,
    3294654272,
    1085426528,
    2410494888,
    3495818564
]


def crop_out_by_keypoints(volume, keypoints_xyz,
                          z_imgs=3,
                          crop_size_h=128,
                          crop_size_w=128,
                          transform=None,
                          resize_to_size=None,
                          G_IMG_SIZE=512):
    sub_volume_list = []
    for p in keypoints_xyz:
        x, y, z = np.round(p[0]), np.round(p[1]), np.round(p[2])
        x, y, z = int(x), int(y), int(z)
        # no z
        if z < 0:
            if z_imgs is not None:
                v = np.zeros((z_imgs, crop_size_h, crop_size_w), dtype=volume.dtype)
            else:
                v = np.zeros((volume.shape[0], crop_size_h, crop_size_w), dtype=volume.dtype)

            sub_volume_list.append(v)
            continue
        bbox = [x - crop_size_w // 2,
                y - crop_size_h // 2,
                x + crop_size_w // 2,
                y + crop_size_h // 2]
        # 如果bbox在边界，偏移它以使其具有正确的img_size大小
        if bbox[0] < 0:
            bbox[2] = crop_size_w
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[3] = crop_size_h
            bbox[1] = 0
        if bbox[2] > G_IMG_SIZE:
            bbox[0] = G_IMG_SIZE - crop_size_w
            bbox[2] = G_IMG_SIZE
        if bbox[3] > G_IMG_SIZE:
            bbox[1] = G_IMG_SIZE - crop_size_h
            bbox[3] = G_IMG_SIZE

        bbox = [int(e) for e in bbox]
        if z_imgs is not None:
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
        else:
            v = volume[:, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
        if transform is not None:
            v = v.transpose(1, 2, 0)
            v = transform(image=v)['image']
            v = v.transpose(2, 0, 1)

        sub_volume_list.append(v)

    volume_crop = np.array(sub_volume_list)

    if resize_to_size is not None:
        volume_crop = torch.from_numpy(volume_crop)
        volume_crop = F.interpolate(volume_crop, (resize_to_size, resize_to_size))
        volume_crop = volume_crop.numpy()

    return volume_crop, None


def warp_img_and_pts(m_512, p_512, affine):
    scale, theta, shift = affine
    theta = theta / 180 * np.pi
    mat = np.array([
        scale * cos(theta), -scale * sin(theta), shift[0],
        scale * sin(theta), scale * cos(theta), shift[1],
    ]).reshape(2, 3)

    p_align = np.concatenate([p_512.reshape(-1, 2), np.ones((5, 1))], axis=1) @ mat.T
    p_align = p_align.reshape(-1, 2)
    # m_align = cv2.warpAffine(m_512, mat, (512, 512))
    for i in range(len(m_512)):
        h, w = m_512[i].shape
        m_512[i] = cv2.warpAffine(m_512[i], mat, (w, h))
    return m_512, p_align


class Sag_T2_Dataset(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None,
                 phase='train', z_imgs=3,
                 img_size=512,
                 crop_size_h=64,
                 crop_size_w=128,
                 resize_to_size=128,
                 use_affine=False,
                 cache_dir='/cache_sag/',
                 hard_list_use_gt_points=True):
        super(Sag_T2_Dataset, self).__init__()
        self.img_size = img_size
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.resize_to_size = resize_to_size

        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Sagittal T2/STIR'])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
        self.phase = phase
        self.z_imgs = z_imgs
        self.hard_list_use_gt_points = hard_list_use_gt_points
        print('[Sag_T2_Dataset] z_imgs: ', z_imgs)

        level2int = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        self.coord_label_df = self.coord_label_df.replace(level2int)
        self.df = df
        self.transform = transform
        if cache_dir is None:
            self.cache_dir = self.data_dir + '/cache_sag/'
        else:
            self.cache_dir = self.data_dir + '/' + cache_dir
        # os.system(f'rm -r {self.cache_dir}')
        create_dir(self.cache_dir)
        self.build_gt_keypoints_info()

        # self.pred_sag_keypoints_infos_3d = pickle.load(
        #     open(f'{data_dir}/pred_keypoints/v2_sag_3d_keypoints_en3.pkl', 'rb'))
        self.pred_sag_keypoints_infos_3d = pickle.load(
            open(f'{data_dir}/v20_sag_T2_3d_keypoints_en5_resnet34d.pkl', 'rb'))
        self.pred_sag_keypoints_infos_3d_oof = pickle.load(
            open(f'{data_dir}/v20_sag_T2_3d_keypoints_oof_resnet34d.pkl', 'rb'))

        # exclude = []
        # for _, row in df.iterrows():
        #     study_id = int(row['study_id'])
        #     if study_id not in self.pred_sag_keypoints_infos_3d:
        #         print(f'[warn] {study_id} not in pred_sag_keypoints_infos_3d')
        #         exclude.append(study_id)
        # self.df = self.df[~self.df['study_id'].isin(exclude)]

        self.use_affine = use_affine
        if use_affine:
            self.saggital_affine_infos = pickle.load(
                open(f'{data_dir}/saggital_affine.pkl', 'rb'))
            self.saggital_t2_rbox_info = build_saggital_t2_rbox(data_dir)

    def build_gt_keypoints_info(self):
        self.gt_keypoints_info = {}
        self.study_id_to_sids = {}
        for _, row in self.df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            if len(series_id_list) == 0:
                print(f'Sag_T2_Dataset [WARN] {study_id} has no series_id')

            self.study_id_to_sids[study_id] = series_id_list
            for series_id in series_id_list:
                if study_id not in self.gt_keypoints_info:
                    self.gt_keypoints_info[study_id] = {}
                pts = self.get_sag_t2_keypoints(study_id, series_id)
                self.gt_keypoints_info[study_id][series_id] = pts

    def get_sag_t2_keypoints(self, study_id, series_id):
        coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
        coord_sub_df = coord_sub_df[
            coord_sub_df['series_id'] == series_id]  # print('coord_sub_df: ', len(coord_sub_df))
        keypoints = -np.ones((5, 1, 3), np.float32)  # 5_level,  x_y_ins_num
        # if len(coord_sub_df) != 5:
        #     print('[WARN] len(coord_sub_df) !=5 study_id: ', study_id, series_id)
        #     print(len(coord_sub_df))
        for _, row in coord_sub_df.iterrows():
            idx = 0
            x, y, instance_number = row['x'], row['y'], row['instance_number']
            keypoints[row['level'], idx, 0] = x
            keypoints[row['level'], idx, 1] = y
            keypoints[row['level'], idx, 2] = instance_number

        keypoints = keypoints.reshape(-1, 3)
        return keypoints

    def load_dicom_and_pts(self, study_id, series_id, img_size=512):
        fn1 = f'{self.cache_dir}/sag_t2_{study_id}_{series_id}_img.npz'
        fn2 = f'{self.cache_dir}/sag_t2_{study_id}_{series_id}_pts.npy'
        if os.path.exists(fn1) and  os.path.exists(fn2):
            arr = np.load(fn1)['arr_0']
            keypoints = np.load(fn2)
            return arr, keypoints

        else:
            dicom_dir = f'{self.data_dir}/train_images/{study_id}/{series_id}/'
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=img_size, )
            keypoints = self.gt_keypoints_info[study_id][series_id]
            keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=img_size)
            np.savez_compressed(fn1, arr)
            np.save(fn2, keypoints)
            return arr, keypoints

    def get_item_by_study_id(self, study_id, item=None):
        if item is None:
            item = self.df[self.df['study_id'] == study_id].iloc[0]
        label = item[1:].values.astype(np.int64)
        sag_t2_label = label[0: 5]
        sids = self.study_id_to_sids[study_id]

        img_size = self.img_size
        crop_size_h = self.crop_size_h
        crop_size_w = self.crop_size_w
        resize_to_size = self.resize_to_size
        if len(sids) == 0:
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            if self.z_imgs is not None:
                s_t2 = np.zeros((5, self.z_imgs, resize_to_size, resize_to_size), dtype=np.float32)
            else:
                s_t2 = np.zeros((5, 10, resize_to_size, resize_to_size), dtype=np.float32)
            sag_t2_label[:] = -100
        else:
            # TODO use multi
            sid = sids[0]
            s_t2, gt_keypoints = self.load_dicom_and_pts(study_id, sid, img_size)
            if self.z_imgs is not None:
                assert s_t2.shape[0] >= self.z_imgs

            c2 = s_t2.shape[0]
            # pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id]['points'].reshape(15, 3)
            # pred_keypoints[:, :2] = img_size * pred_keypoints[:, :2] / 4.0
            # pred_keypoints[:5, 2] = c2 * pred_keypoints[:5, 2] / 16.0

            pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id][sid].reshape(5, 3)
            if self.phase == 'train' and random.random() < 0.5:
                pred_keypoints = self.pred_sag_keypoints_infos_3d_oof[study_id][sid].reshape(5, 3)
            pred_keypoints[:, :2] = img_size * pred_keypoints[:, :2]
            pred_keypoints[:, 2] = c2 * pred_keypoints[:, 2]


            pred_keypoints = np.round(pred_keypoints).astype(np.int64)
            if self.phase == 'train' and random.random() < 0.5:
                t2_keypoints = gt_keypoints
            else:
                t2_keypoints = pred_keypoints[:5]

            if study_id in sag_z_hard_list:
                t2_keypoints = gt_keypoints

            if self.hard_list_use_gt_points:
                if self.phase != 'train':
                    if study_id in hard_sag_t2_keypoints_study_id_list:
                        t2_keypoints = gt_keypoints
                    if study_id in hard_sag_t2_z_keypoints_study_id_list:
                        t2_keypoints = gt_keypoints

                if self.phase == 'train' and random.random() < 0.5:
                    if study_id in hard_sag_t2_keypoints_study_id_list:
                        t2_keypoints = gt_keypoints
                    if study_id in hard_sag_t2_z_keypoints_study_id_list:
                        t2_keypoints = gt_keypoints

            #t2_keypoints = gt_keypoints
            #t2_keypoints = pred_keypoints

            if self.use_affine:
                rbox_keypoints = self.saggital_t2_rbox_info[study_id][sid]['keypoints']
                t2_keypoints[:, :2] = rbox_keypoints[:, 1, :]
                affine = self.saggital_affine_infos[study_id][sid]['affine']
                s_t2, t2_keypoints[:, :2] = warp_img_and_pts(s_t2, t2_keypoints[:, :2], affine)

            keypoints_xy = t2_keypoints[:, :2]
            keypoints_z = t2_keypoints[:, 2:]
            mask_xy = np.where(keypoints_xy < 0, 0, 1)
            mask_z = np.where(keypoints_z < 0, 0, 1)

            for i in range(5):
                if keypoints_z[i] < 0:
                    sag_t2_label[i] = -100

            if self.phase == 'train' and random.random() < 0.5:
                noise1 = np.random.uniform(-5.0, 5.0, keypoints_xy.shape)
                keypoints_xy = keypoints_xy + noise1

            if self.transform is not None:
                # to h,w,c
                s_t2 = s_t2.transpose(1, 2, 0)
                augmented = self.transform(image=s_t2, keypoints=keypoints_xy)
                s_t2 = augmented['image']
                keypoints_xy = augmented['keypoints']
                keypoints_xy = np.array(keypoints_xy)
                s_t2 = s_t2.transpose(2, 0, 1)

            keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)
            mask = np.concatenate((mask_xy, mask_z), axis=1)
            s_t2_keypoints = keypoints

            # print('study_id: ',study_id)
            # print(s_t2.mean(), s_t2.std())
            # print('s_t2_keypoints: ', s_t2_keypoints)
            # exit(0)
            s_t2, _ = crop_out_by_keypoints(s_t2, s_t2_keypoints,
                                            z_imgs=self.z_imgs,
                                            crop_size_h=crop_size_h,
                                            crop_size_w=crop_size_w,
                                            resize_to_size=resize_to_size,
                                            G_IMG_SIZE=img_size)

        return {
            's_t2': s_t2,
            'sag_t2_label': sag_t2_label
        }

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        return self.get_item_by_study_id(study_id, item)


class Sag_T1_Dataset(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None,
                 phase='train', z_imgs=3,
                 img_size=512,
                 crop_size_h=64,
                 crop_size_w=128,
                 resize_to_size=128,
                 cache_dir='/cache_sag/',
                 hard_list_use_gt_points=False):
        super(Sag_T1_Dataset, self).__init__()
        self.img_size = img_size
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.resize_to_size = resize_to_size

        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Sagittal T1'])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
        self.phase = phase
        self.z_imgs = z_imgs
        self.hard_list_use_gt_points = hard_list_use_gt_points
        print('[Sag_T1_Dataset] z_imgs: ', z_imgs)

        level2int = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        self.coord_label_df = self.coord_label_df.replace(level2int)
        self.df = df
        self.transform = transform
        if cache_dir is None:
            self.cache_dir = self.data_dir + '/cache_sag/'
        else:
            self.cache_dir = self.data_dir + '/' + cache_dir
        create_dir(self.cache_dir)
        self.build_gt_keypoints_info()

        # self.pred_sag_keypoints_infos_3d = pickle.load(
        #     open(f'{data_dir}/pred_keypoints/v2_sag_3d_keypoints_en3.pkl', 'rb'))

        self.pred_sag_keypoints_infos_3d = pickle.load(
            open(f'{data_dir}/v20_sag_T1_3d_keypoints_en5_resnet34d.pkl', 'rb'))
        self.pred_sag_keypoints_infos_3d_oof = pickle.load(
            open(f'{data_dir}/v20_sag_T1_3d_keypoints_oof_resnet34d.pkl', 'rb'))

    def build_gt_keypoints_info(self):
        self.gt_keypoints_info = {}
        self.study_id_to_sids = {}
        for _, row in self.df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            self.study_id_to_sids[study_id] = series_id_list
            if len(series_id_list) == 0:
                print(f'Sag_T1_Dataset [WARN] {study_id} has no series_id')
            for series_id in series_id_list:
                if study_id not in self.gt_keypoints_info:
                    self.gt_keypoints_info[study_id] = {}
                pts = self.get_sag_t1_keypoints(study_id, series_id)
                self.gt_keypoints_info[study_id][series_id] = pts

    def get_sag_t1_keypoints(self, study_id, series_id):
        coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
        coord_sub_df = coord_sub_df[
            coord_sub_df['series_id'] == series_id]  # print('coord_sub_df: ', len(coord_sub_df))
        keypoints = -np.ones((5, 2, 3), np.float32)  # 5_level,  x_y_ins_num
        # if len(coord_sub_df) != 10:
        #     print('[WARN] len(coord_sub_df) !=10 study_id: ', study_id, series_id)
        #     print(len(coord_sub_df))

        for _, row in coord_sub_df.iterrows():
            idx = 0
            if row['condition'] == 'Right Neural Foraminal Narrowing':
                idx = 1
            x, y, instance_number = row['x'], row['y'], row['instance_number']
            keypoints[row['level'], idx, 0] = x
            keypoints[row['level'], idx, 1] = y
            keypoints[row['level'], idx, 2] = instance_number

        keypoints = keypoints.reshape(-1, 3)
        return keypoints

    def load_dicom_and_pts(self, study_id, series_id, img_size=512):
        fn1 = f'{self.cache_dir}/sag_t1_{study_id}_{series_id}_img.npz'
        fn2 = f'{self.cache_dir}/sag_t1_{study_id}_{series_id}_pts.npy'
        if os.path.exists(fn1) and  os.path.exists(fn2):
            arr = np.load(fn1)['arr_0']
            keypoints = np.load(fn2)
            return arr, keypoints

        else:
            dicom_dir = f'{self.data_dir}/train_images/{study_id}/{series_id}/'
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=img_size, )
            keypoints = self.gt_keypoints_info[study_id][series_id]
            keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=img_size)
            np.savez_compressed(fn1, arr)
            np.save(fn2, keypoints)
            return arr, keypoints

    def get_item_by_study_id(self, study_id, item=None):
        if item is None:
            item = self.df[self.df['study_id'] == study_id].iloc[0]
        label = item[1:].values.astype(np.int64)
        sag_t1_label = label[5: 15]
        sids = self.study_id_to_sids[study_id]

        img_size = self.img_size
        crop_size_h = self.crop_size_h
        crop_size_w = self.crop_size_w
        resize_to_size = self.resize_to_size
        if len(sids) == 0:
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            if self.z_imgs is not None:
                s_t1 = np.zeros((10, self.z_imgs, resize_to_size, resize_to_size), dtype=np.float32)
            else:
                s_t1 = np.zeros((10, 10, resize_to_size, resize_to_size), dtype=np.float32)
            sag_t1_label[:] = -100
        else:
            # TODO use multi
            sid = sids[0]
            s_t1, gt_keypoints = self.load_dicom_and_pts(study_id, sid, img_size)
            gt_keypoints = gt_keypoints.reshape(5, 2, 3)
            gt_keypoints = gt_keypoints.transpose(1, 0, 2).reshape(-1, 3)
            # print(gt_keypoints)
            if self.z_imgs is not None:
                assert s_t1.shape[0] >= self.z_imgs

            c1 = s_t1.shape[0]
            # pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id]['points'].reshape(15, 3)
            # pred_keypoints[:, :2] = img_size * pred_keypoints[:, :2] / 4.0
            # pred_keypoints[5:, 2] = c1 * pred_keypoints[5:, 2] / 16.0

            pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id][sid].reshape(10, 3)
            if self.phase == 'train' and random.random() < 0.5:
                pred_keypoints = self.pred_sag_keypoints_infos_3d_oof[study_id][sid].reshape(10, 3)

            pred_keypoints[:, :2] = img_size * pred_keypoints[:, :2]
            pred_keypoints[:, 2] = c1 * pred_keypoints[:, 2]

            pred_keypoints = np.round(pred_keypoints).astype(np.int64)

            if self.phase == 'train' and random.random() < 0.5:
                t1_keypoints = gt_keypoints
            else:
                t1_keypoints = pred_keypoints
                #t1_keypoints = pred_keypoints[5:]

            if study_id in sag_z_hard_list:
                t1_keypoints = gt_keypoints

            if self.hard_list_use_gt_points:
                if self.phase != 'train':
                    if study_id in hard_sag_t1_keypoints_study_id_list:
                        t1_keypoints = gt_keypoints
                    if study_id in hard_sag_t1_z_keypoints_study_id_list:
                        t1_keypoints = gt_keypoints

                if self.phase == 'train' and random.random() < 0.5:
                    if study_id in hard_sag_t1_keypoints_study_id_list:
                        t1_keypoints = gt_keypoints
                    if study_id in hard_sag_t1_z_keypoints_study_id_list:
                        t1_keypoints = gt_keypoints

            #t1_keypoints = gt_keypoints
            #t1_keypoints = pred_keypoints

            keypoints_xy = t1_keypoints[:, :2]
            keypoints_z = t1_keypoints[:, 2:]
            mask_xy = np.where(keypoints_xy < 0, 0, 1)
            mask_z = np.where(keypoints_z < 0, 0, 1)

            for i in range(10):
                if keypoints_z[i] < 0:
                    sag_t1_label[i] = -100

            if self.phase == 'train' and random.random() < 0.5:
                noise1 = np.random.uniform(-5.0, 5.0, keypoints_xy.shape)
                keypoints_xy = keypoints_xy + noise1

            if self.transform is not None:
                # to h,w,c
                s_t1 = s_t1.transpose(1, 2, 0)
                augmented = self.transform(image=s_t1, keypoints=keypoints_xy)
                s_t1 = augmented['image']
                keypoints_xy = augmented['keypoints']
                keypoints_xy = np.array(keypoints_xy)
                s_t1 = s_t1.transpose(2, 0, 1)

            keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)

            mask = np.concatenate((mask_xy, mask_z), axis=1)

            s_t1_left_keypoints = keypoints[:5]
            s_t1_right_keypoints = keypoints[5:]

            s_t1_left, _ = crop_out_by_keypoints(s_t1, s_t1_left_keypoints,
                                                 z_imgs=self.z_imgs,
                                                 crop_size_h=crop_size_h,
                                                 crop_size_w=crop_size_w,
                                                 resize_to_size=resize_to_size,
                                                 G_IMG_SIZE=img_size)
            #
            s_t1_right, _ = crop_out_by_keypoints(s_t1, s_t1_right_keypoints,
                                                  z_imgs=self.z_imgs,
                                                  crop_size_h=crop_size_h,
                                                  crop_size_w=crop_size_w,
                                                  resize_to_size=resize_to_size,
                                                  G_IMG_SIZE=img_size)
            s_t1 = np.concatenate([s_t1_left, s_t1_right], axis=0)

        return {
            's_t1': s_t1,
            'sag_t1_label': sag_t1_label
        }

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        return self.get_item_by_study_id(study_id, item)


def sort_points(pts):
    sorted_pts = sorted(pts, key=lambda pt: pt[0])
    left_pts = sorted(sorted_pts[:2], key=lambda pt: pt[1])
    right_pts = sorted(sorted_pts[2:], key=lambda pt: pt[1])

    return [left_pts[0], right_pts[0], right_pts[1], left_pts[1]]


def build_saggital_t2_rbox(data_root, img_size=512):
    df = pd.read_csv(f'{data_root}/coords_rsna_improved_saggital_t2.csv')
    df = df.sort_values(['series_id', 'level'])
    print(df['level'].head(10))
    d = df.groupby("series_id")[["relative_x",
                                 "relative_y",
                                 "study_id",
                                 "series_id"
                                 ]].apply(
        lambda x: list(x.itertuples(index=False, name=None)))
    saggital_t2_rbox_info = {}
    for k, v in d.items():
        v = np.array(v)
        relative_xy = v[:, :2].reshape(5, 2, 2)
        study_id = int(v[0, 2])
        series_id = int(v[0, 3])
        keypoints = img_size * relative_xy
        pts1 = keypoints[:, 0]
        pts2 = keypoints[:, 1]
        distances = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
        w = np.mean(distances)
        rboxes = []

        for p in keypoints:
            p0 = p[0]
            p1 = p[1]
            center = p1
            angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            h = w / 2
            rotated_rect = ((center[0], center[1]), (w, h), angle)
            box = cv2.boxPoints(rotated_rect)
            box = box.tolist()
            box = sort_points(box)
            box = np.array(box, dtype='float32')
            rboxes.append(box)
        rboxes = np.array(rboxes)
        if study_id not in saggital_t2_rbox_info.keys():
            saggital_t2_rbox_info[study_id] = {}
        saggital_t2_rbox_info[study_id][series_id] = {
            "rboxes": rboxes,
            "keypoints": keypoints,
            "w": w,
            "h": w / 2,
        }

    return saggital_t2_rbox_info


class Sag_T2_T1_Dataset(Dataset):
    def __init__(self, data_dir,
                 df: pd.DataFrame,
                 transform=None,
                 phase='train',
                 z_imgs=3,
                 # img_size=640,
                 # crop_size_h=80,
                 # crop_size_w=160,
                 # resize_to_size=160,
                 img_size=512,
                 crop_size_h=64,
                 crop_size_w=128,
                 resize_to_size=128
                 ):
        # without = [
        #     490052995,
        #     1261271580,
        #     2507107985,
        #     2626030939,
        #     2773343225,
        #     3109648055,
        #     3387993595,
        #     2492114990, 3008676218, 2780132468, 3637444890
        # ]
        #self.df = df[~df['study_id'].isin(without)]
        self.df = df
        self.sag_t2_dset = Sag_T2_Dataset(data_dir,
                                          self.df, transform=transform,
                                          phase=phase,
                                          z_imgs=z_imgs,
                                          img_size=img_size,
                                          crop_size_h=crop_size_h,
                                          crop_size_w=crop_size_w,
                                          resize_to_size=resize_to_size,
                                          use_affine=False,
                                          cache_dir='/cache_sag/')
        self.sag_t1_dset = Sag_T1_Dataset(data_dir,
                                          self.df, transform=transform,
                                          phase=phase,
                                          z_imgs=z_imgs,
                                          img_size=img_size,
                                          crop_size_h=crop_size_h,
                                          crop_size_w=crop_size_w,
                                          resize_to_size=resize_to_size,
                                          cache_dir='/cache_sag/'
                                          )
        print('[Sag_T2_T1_Dataset] samples: ', len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        t2_dict = self.sag_t2_dset.get_item_by_study_id(study_id, item)
        s_t2 = t2_dict['s_t2']
        sag_t2_label = t2_dict['sag_t2_label']
        t1_dict = self.sag_t1_dset.get_item_by_study_id(study_id, item)
        s_t1 = t1_dict['s_t1']
        sag_t1_label = t1_dict['sag_t1_label']

        label = np.concatenate((sag_t2_label, sag_t1_label), axis=0)
        img = np.concatenate((s_t2, s_t1), axis=0)
        ret_dict = {}

        cond = [0] * 5 + [1] * 5 + [2] * 5
        cond = np.array(cond, dtype=np.int64)
        # ret_dict['s_t2'] = s_t2
        # ret_dict['s_t1'] = s_t1
        ret_dict['cond'] = cond
        ret_dict['label'] = label
        ret_dict['img'] = img
        ret_dict['study_id'] = study_id
        return ret_dict


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
    df = df[df['study_id'] == 4232806580]
    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    create_dir(debug_dir)
    import albumentations as A

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = Sag_T2_T1_Dataset(data_root, df, transform=transforms_train,
                             phase='train', z_imgs=3)

    for d in dset:
        img = d['img']
        print('img: ', img.shape)
        print(d['label'])
        cv2.imwrite(f'{debug_dir}/01_v24_sag_t2_t1.jpg', convert_to_cv2_img(img))
        break
