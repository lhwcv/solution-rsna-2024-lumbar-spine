# -*- coding: utf-8 -*-
import os
import random

import scipy
import timm
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F
from src.utils.comm import create_dir
import torch.nn as nn
import albumentations as A

from train_scripts.v20.dicom import load_dicom, rescale_keypoints_by_meta
from train_scripts.v20.hard_list import hard_sag_t2_keypoints_study_id_list, hard_sag_t1_z_keypoints_study_id_list, \
    hard_sag_t2_z_keypoints_study_id_list, hard_sag_t1_keypoints_study_id_list


class Sag_2D_Point_Dataset(Dataset):
    def __init__(self, data_dir,
                 df: pd.DataFrame,
                 transform=None,
                 phase='train',
                 series_description='Sagittal T2/STIR',
                 img_size=512,
                 exclude_hard = False,
                 ):
        super(Sag_2D_Point_Dataset, self).__init__()
        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin([series_description])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
        level2int = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        self.coord_label_df = self.coord_label_df.replace(level2int)
        self.transform = transform
        self.phase = phase
        self.img_size = img_size

        exclude_study_id_list = []
        if series_description == 'Sagittal T2/STIR':
            exclude_study_id_list = hard_sag_t2_keypoints_study_id_list

        if series_description == 'Sagittal T1':
            exclude_study_id_list = hard_sag_t1_keypoints_study_id_list

        if self.phase in ['train', 'valid'] and exclude_hard:
            print('exclude_study_id_list len: ', len(exclude_study_id_list))
            self.df = df[~df['study_id'].isin(exclude_study_id_list)]
        else:
            self.df = df

        self.samples = []

        for _, row in self.df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()

            for sid in series_id_list:
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == sid]
                for _, ins_num_g in coord_sub_df.groupby('instance_number'):
                    keypoints = -np.ones((5, 3))
                    ins_num = ins_num_g['instance_number'].iloc[0]
                    for _, row in ins_num_g.iterrows():
                        x, y, instance_number = row['x'], row['y'], row['instance_number']
                        keypoints[row['level'], 0] = x
                        keypoints[row['level'], 1] = y
                        keypoints[row['level'], 2] = instance_number
                    self.samples.append({
                        'study_id': study_id,
                        'series_id': sid,
                        'keypoints': keypoints,
                        'ins_num': ins_num
                    })
        print('samples: ', len(self.samples))
        self.cache_dir = self.data_dir + '/cache/'
        create_dir(self.cache_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        study_id = sample['study_id']
        series_id = sample['series_id']
        ins_num = sample['ins_num']
        fn1 = f'{self.cache_dir}/sag_2d_keypoints_{study_id}_{series_id}_{ins_num}_img.npz'
        fn2 = f'{self.cache_dir}/sag_2d_keypoints_{study_id}_{series_id}_{ins_num}_pts.npy'
        if os.path.exists(fn1):
            img = np.load(fn1)['arr_0']
            keypoints = np.load(fn2)
        else:
            keypoints = sample['keypoints']
            dicom_dir = f'{self.data_dir}/train_images/{study_id}/{series_id}/'
            base_size = self.img_size
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=self.img_size, )
            keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=base_size)
            dicom_instance_numbers_to_idx = meta['dicom_instance_numbers_to_idx']
            idx = dicom_instance_numbers_to_idx[ins_num]
            img = arr[idx]
            np.savez_compressed(fn1, img)
            np.save(fn2, keypoints)

        keypoints = keypoints[:, :2]
        mask = np.where(keypoints < 0, 0, 1)
        img = img[np.newaxis, :, :]
        if self.transform is not None:
            # to h,w,c
            img = img.transpose(1, 2, 0)
            augmented = self.transform(image=img, keypoints=keypoints)
            img = augmented['image']
            keypoints = augmented['keypoints']
            keypoints = np.array(keypoints)
            img = img.transpose(2, 0, 1)

        keypoints = keypoints.astype(np.float32)
        keypoints = keypoints / self.img_size
        return {
            'imgs': img,
            'keypoints': keypoints.reshape(-1),
            'mask': mask.reshape(-1)
        }


class Sag_3D_Point_Dataset(Dataset):
    def __init__(self, data_dir,
                 df: pd.DataFrame,
                 transform=None,
                 phase='train',
                 series_description='Sagittal T2/STIR',
                 img_size=512,
                 depth_3d=32,
                 img_size_3d=384,
                 exclude_hard = False,
                 ):
        super(Sag_3D_Point_Dataset, self).__init__()
        self.depth_3d = depth_3d
        self.img_size_3d = img_size_3d
        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin([series_description])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
        level2int = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        self.coord_label_df = self.coord_label_df.replace(level2int)
        self.transform = transform
        self.phase = phase
        self.img_size = img_size

        exclude_study_id_list = []
        if series_description == 'Sagittal T2/STIR' and exclude_hard:
            exclude_study_id_list = hard_sag_t2_keypoints_study_id_list + \
                                    hard_sag_t2_z_keypoints_study_id_list

        if series_description == 'Sagittal T1' and exclude_hard:
            exclude_study_id_list = hard_sag_t1_keypoints_study_id_list + \
                                    hard_sag_t1_z_keypoints_study_id_list

        if self.phase in ['train', 'valid']:
            print('exclude_study_id_list len: ', len(exclude_study_id_list))
            self.df = df[~df['study_id'].isin(exclude_study_id_list)]
        else:
            self.df = df
        self.series_description = series_description
        self.samples = []

        for _, row in self.df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            if len(series_id_list) == 0:
                print(f'[WARN] {study_id} has no series_id for {series_description}')

            for sid in series_id_list:
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == sid]
                if series_description == 'Sagittal T2/STIR':
                    keypoints = -np.ones((5, 1, 3), dtype=np.float32)
                else:
                    keypoints = -np.ones((5, 2, 3), dtype=np.float32)

                for _, row in coord_sub_df.iterrows():
                    idx = 0
                    if row['condition'] == 'Right Neural Foraminal Narrowing':
                        idx = 1
                    x, y, instance_number = row['x'], row['y'], row['instance_number']
                    keypoints[row['level'], idx, 0] = x
                    keypoints[row['level'], idx, 1] = y
                    keypoints[row['level'], idx, 2] = instance_number

                keypoints = keypoints.transpose(1, 0, 2)
                self.samples.append({
                    'study_id': study_id,
                    'series_id': sid,
                    'keypoints': keypoints.reshape(-1, 3),
                })
        print('samples: ', len(self.samples))
        self.cache_dir = self.data_dir + '/cache/'
        create_dir(self.cache_dir)

    def __len__(self):
        return len(self.samples)

    def mean_z(self, pts):
        z = 0
        n = 0
        for p in pts:
            if p[2] > 0:
                z += p[2]
                n += 1
        if n == 0:
            return -1
        return z / n

    def __getitem__(self, idx):
        sample = self.samples[idx]
        study_id = sample['study_id']
        series_id = sample['series_id']

        fn1 = f'{self.cache_dir}/sag_3d_keypoints_{study_id}_{series_id}_img.npz'
        fn2 = f'{self.cache_dir}/sag_3d_keypoints_{study_id}_{series_id}_pts.npy'
        if os.path.exists(fn1):
            arr = np.load(fn1)['arr_0']
            keypoints = np.load(fn2)
        else:
            keypoints = sample['keypoints']
            dicom_dir = f'{self.data_dir}/train_images/{study_id}/{series_id}/'
            base_size = self.img_size
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=self.img_size, )
            keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=base_size)
            np.savez_compressed(fn1, arr)
            np.save(fn2, keypoints)
        # keypoints: 5 or 10, 3
        if self.series_description == 'Sagittal T1':
            pts = keypoints.reshape(2, 5, 3)
            z_left = self.mean_z(pts[0])
            z_right = self.mean_z(pts[1])
            if z_left < z_right and z_left != -1:
                print('[WARN] z_left < z_right, study_id: ', study_id, series_id)
                print(z_left, z_right)
                dicom_dir = f'{self.data_dir}/train_images/{study_id}/{series_id}/'
                arr, meta = load_dicom(dicom_dir,
                                       plane='sagittal', reverse_sort=False,
                                       img_size=self.img_size, )
                print('PatientPosition: ', meta['PatientPosition'])

        keypoints_xy = keypoints[:, :2]
        keypoints_z = keypoints[:, 2:]

        mask_xy = np.where(keypoints_xy < 0, 0, 1)
        mask_z = np.where(keypoints_z < 0, 0, 1)
        keypoints_xy = keypoints_xy
        keypoints_z = keypoints_z / arr.shape[0]

        if self.transform is not None:
            # to h,w,c
            arr = arr.transpose(1, 2, 0)
            augmented = self.transform(image=arr, keypoints=keypoints_xy)
            arr = augmented['image']
            keypoints_xy = augmented['keypoints']
            keypoints_xy = np.array(keypoints_xy)
            arr = arr.transpose(2, 0, 1)

        origin_depth = arr.shape[0]
        arr = arr.astype(np.float32)
        arr = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        arr = F.interpolate(arr, size=(self.depth_3d, self.img_size_3d, self.img_size_3d)).squeeze(0)

        keypoints_xy[:, :2] = keypoints_xy[:, :2] / self.img_size

        keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)
        mask = np.concatenate((mask_xy, mask_z), axis=1)

        return {
            'imgs': arr,
            'keypoints': keypoints.reshape(-1),
            'mask': mask.reshape(-1),
            'study_id': study_id,
            'series_id': series_id,
            'origin_depth': origin_depth,
        }


def test_sag_2d_keypoints_data():
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

    dset = Sag_2D_Point_Dataset(data_root, df, transform=transforms_train,
                                phase='train', series_description='Sagittal T1')

    for d in dset:
        img = d['imgs'][0]
        keypoints = d['keypoints'].reshape(5, 2) * 512
        keypoints = np.asarray(keypoints, np.int64)
        mask = d['mask'].reshape(5, 2)
        print(keypoints)
        print(mask)
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for p in keypoints:
            x, y = p[0], p[1]
            img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)

        cv2.imwrite(f'{debug_dir}/0_v20_sag_t1_2d_keypoints.jpg', img)
        break


def test_sag_3d_keypoint_data():
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    #study_ids = [1085426528, 2135829458]
    #df = df[df['study_id'].isin(study_ids)]
    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    create_dir(debug_dir)
    import tqdm

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = Sag_3D_Point_Dataset(data_root, df, transform=None,
                                phase='train', series_description='Sagittal T1')
    dloader = DataLoader(dset, num_workers=12)
    print(len(dloader))
    for d in tqdm.tqdm(dloader):
        #continue
        volumn = d['imgs'][0][0]
        print('img: ', volumn.shape)
        keypoints = d['keypoints'][0].reshape(-1, 3)
        print(keypoints)
        keypoints[:, :2] = keypoints[:, :2] * 384
        keypoints[:, 2] = keypoints[:, 2] * volumn.shape[0]

        print('mask: ', d['mask'][0])
        imgs = []
        for i, p in enumerate(keypoints):
            x, y, z = int(p[0]), int(p[1]), int(p[2])
            if z < 0:
                continue
            print('xyz: ', x, y, z)
            img = volumn[z]

            img = 255 * (img - img.min()) / (img.max() - img.min())
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
            cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            la = i % 5 + 1
            cv2.putText(img, 'level: ' + str(la), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            imgs.append(img)

        img_concat = np.concatenate(imgs, axis=0)
        cv2.imwrite(f'{debug_dir}/0_v20_sag_t1_3d_keypoints.jpg', img_concat)
        exit(0)


if __name__ == '__main__':
    test_sag_3d_keypoint_data()
    # test_sag_2d_keypoints_data()
