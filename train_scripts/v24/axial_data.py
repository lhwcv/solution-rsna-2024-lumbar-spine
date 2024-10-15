# -*- coding: utf-8 -*-
import os
import random

import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F
from src.utils.comm import create_dir
import albumentations as A

# window_center , window_width, intercept, slope = get_windowing(data)
# data.pixel_array = data.pixel_array * slope + intercept
# min_value = window_center - window_width // 2
# max_value = window_center + window_width // 2
# data.pixel_array.clip(min_value, max_value, out=data.pixel_array)
from train_scripts.v20.dicom import load_dicom, rescale_keypoints_by_meta
from train_scripts.v24.axial_model import Axial_Level_Cls_Model
from train_scripts.v24.sag_data import Sag_T2_Dataset

hard_axial_study_id_list = [391103067,
                            953639220,
                            2460381798,
                            2690161683,
                            3650821463,
                            3949892272,
                            677672203,  # 左右点标注反了
                            ]


def _get_keypoint_mean_margin(pts):
    # pts 5, 3
    mean_margin = []
    for i in range(4):
        z0 = pts[i, 2]
        z1 = pts[i + 1, 2]
        if z0 != -1 and z1 != -1:
            z = abs(z1 - z0)
            mean_margin.append(z)
    if len(mean_margin) == 0:
        mean_margin = -1
    else:
        mean_margin = np.mean(mean_margin)
    return mean_margin


def _gen_label_by_keypoints(pts, mean_margin, z_len,
                            label=None, sparse_label=None):
    # pts (5, 3)
    if label is None:
        label = -100 * np.ones(z_len)
    if sparse_label is None:
        sparse_label = np.zeros((z_len, 5), dtype=np.float32)
        sparse_label[:] = 0.001

    # margin = min(mean_margin / 4, 2)
    margin = 2
    for level, p in enumerate(pts):
        z = p[2]
        if z < 0:
            assert z == -1
            continue
        start_idx = int(np.round(z - margin))
        if start_idx < 0:
            start_idx = 0
        end_idx = int(np.round(z + margin)) + 1
        if end_idx > z_len:
            end_idx = z_len
        # if level == 0:
        #     print(start_idx, end_idx)
        label[start_idx:end_idx] = level
        d = max(end_idx - z, z - start_idx)
        assert d != 0
        slope = 0.5 / d

        for i in range(start_idx, end_idx):
            dis = abs(i - z)
            s = 1.0 - dis * slope
            sparse_label[i, level] = s

    return label, sparse_label


def check_label(label_):
    label = label_[label_ != -100].reshape(-1).tolist()
    label_unique = []
    for la in label:
        if la not in label_unique:
            if len(label_unique) > 0:
                if la < label_unique[len(label_unique) - 1]:
                    return False
            label_unique.append(la)
    return True


#
# class Axial_Level_Dataset(Dataset):
#     def __init__(self, data_dir, df: pd.DataFrame, transform=None):
#         super(Axial_Level_Dataset, self).__init__()
#         desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
#         self.image_dir = f"{data_dir}/train_images/"
#         self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]
#         self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
#         level2int = {
#             'L1/L2': 0,
#             'L2/L3': 1,
#             'L3/L4': 2,
#             'L4/L5': 3,
#             'L5/S1': 4
#         }
#         self.coord_label_df = self.coord_label_df.replace(level2int)
#         self.df = df
#         self.transform = transform
#         self.samples = []
#         hard_axial_series_id_list = [1771893480,
#                                      3014494618,
#                                      2114122300,
#                                      2714421343,
#                                      2693084890,
#                                      2444210364
#                                      ]
#         for _, row in df.iterrows():
#             study_id = row['study_id']
#             label = row[1:].values.astype(np.int64)
#             label = label[15:].reshape(2, 5)
#             g = self.desc_df[self.desc_df['study_id'] == study_id]
#             series_id_list = g['series_id'].to_list()
#             if len(series_id_list) != 1:
#                 continue
#
#             for series_id in series_id_list:
#                 if series_id in hard_axial_series_id_list:
#                     continue
#                 coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
#                 coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
#                 if len(coord_sub_df) > 0:
#                     self.samples.append({
#                         'study_id': study_id,
#                         'series_id': series_id,
#                         'label': label
#                     })
#
#         print('samples: ', len(self.samples))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def get_axial_coord(self, study_id, series_id):
#         coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
#         coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
#         keypoints = -np.ones((5, 2, 3), np.float32)  # 5_level, left/right, xyz
#
#         for _, row in coord_sub_df.iterrows():
#             idx = 0
#             if row['condition'] == 'Right Subarticular Stenosis':
#                 idx = 1
#             x, y, z = row['x'], row['y'], row['instance_number']
#
#             keypoints[row['level'], idx, 0] = x
#             keypoints[row['level'], idx, 1] = y
#             keypoints[row['level'], idx, 2] = z
#
#         keypoints = keypoints.reshape(-1, 3)
#         return keypoints
#
#     def __getitem__(self, idx):
#         item = self.samples[idx]
#         study_id = item['study_id']
#         series_id = item['series_id']
#         cls_label = item['label']
#         img_size = 320
#         keypoints = self.get_axial_coord(study_id, series_id)
#         # print(keypoints)
#         dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
#         arr, meta = load_dicom(dicom_dir, img_size=512)
#
#         PatientPosition = meta['PatientPosition']
#         # if PatientPosition == "FFS":
#         #     arr = np.flip(arr, -1)
#         #     arr = np.ascontiguousarray(arr)
#
#         arr = arr.transpose(1, 2, 0)
#         arr = A.center_crop(arr, 384, 384)
#         arr = arr.transpose(2, 0, 1)
#
#         keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=img_size)
#         # print('rescaled: ', keypoints)
#         # print('PatientPosition: ', meta['PatientPosition'])
#         # print('arr shape: ', arr.shape)
#         # resample by SpacingBetweenSlices
#         SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]
#         target_spacing = 3.0
#         z_scale = SpacingBetweenSlices / target_spacing
#         depth = int(arr.shape[0] * z_scale)
#
#         # target_size = (depth, img_size, img_size)
#         target_size = (depth, img_size, img_size)
#         arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
#                             size=target_size).numpy().squeeze()
#
#         # keypoints[:, 2] = keypoints[:, 2] * z_scale
#         for i in range(keypoints.shape[0]):
#             if keypoints[i, 2] != -1:
#                 keypoints[i, 2] = keypoints[i, 2] * z_scale
#
#         keypoints = keypoints.reshape(5, 2, 3)
#         left_keypoints = keypoints[:, 0]
#         right_keypoints = keypoints[:, 1]
#         mean_margin_left = _get_keypoint_mean_margin(left_keypoints)
#         mean_margin_right = _get_keypoint_mean_margin(right_keypoints)
#
#         z_len = arr.shape[0]
#         if mean_margin_left != -1 and mean_margin_right != -1:
#             mean_margin = (mean_margin_right + mean_margin_left) / 2
#         elif mean_margin_left == -1 and mean_margin_right != -1:
#             mean_margin = mean_margin_right
#         elif mean_margin_right == -1 and mean_margin_left != -1:
#             mean_margin = mean_margin_left
#         else:
#             mean_margin = 9 * target_spacing / 4.0
#
#         label, sparse_label_left = _gen_label_by_keypoints(left_keypoints,
#                                                            mean_margin, z_len)
#         label, sparse_label_right = _gen_label_by_keypoints(right_keypoints,
#                                                             mean_margin, z_len, label)
#         sparse_label = (sparse_label_left + sparse_label_right) / 2
#         if self.transform is not None:
#             arr = arr.transpose(1, 2, 0)
#             arr = self.transform(image=arr)['image']
#             arr = arr.transpose(2, 0, 1)
#         # print(study_id, label)
#         if not check_label(label):
#             print('!!!!!')
#             print('study_id: ', study_id)
#             print('sid: ', series_id)
#             coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
#             coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
#             for _, row in coord_sub_df.iterrows():
#                 print(row)
#             print('label: ', label)
#             print('keypoints: ', keypoints)
#
#         for i in range(5):
#             if left_keypoints[i, 2] < 0:
#                 cls_label[0, i] = -100
#             if right_keypoints[i, 2] < 0:
#                 cls_label[1, i] = -100
#         cls_label = cls_label.reshape(-1)
#
#         return {
#             'img': arr.astype(np.float32),
#             'label': label.astype(np.int64),
#             'sparse_label': sparse_label,
#             'cls_label': cls_label,
#
#             'study_id': study_id,
#             'series_id': series_id,
#             'PatientPosition': meta['PatientPosition']
#         }


class Axial_2D_Point_Dataset(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None,
                 phase='train'):
        super(Axial_2D_Point_Dataset, self).__init__()
        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")

        self.df = df
        self.transform = transform
        self.samples = []

        exclude_study_id_list = [
            677672203,  # 左右点标注反了
        ]
        if phase == 'train':
            self.df = df[~df['study_id'].isin(exclude_study_id_list)]
        else:
            self.df = df

        self.samples = []

        for _, row in df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            # if len(series_id_list) != 2:
            #     continue

            # study_ids.append(study_id)
            for sid in series_id_list:
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == sid]
                for _, ins_num_g in coord_sub_df.groupby('instance_number'):
                    pts = -np.ones((2, 3))
                    ins_num = ins_num_g['instance_number'].iloc[0]
                    for _, row in ins_num_g.iterrows():
                        idx = 0
                        if row['condition'] == 'Right Subarticular Stenosis':
                            idx = 1
                        x, y = row['x'], row['y']
                        pts[idx, 0] = x
                        pts[idx, 1] = y
                        pts[idx, 2] = ins_num
                    self.samples.append({
                        'study_id': study_id,
                        'series_id': sid,
                        'keypoints': pts,
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
        fn1 = f'{self.cache_dir}/axial_2d_keypoints_{study_id}_{series_id}_{ins_num}_img.npz'
        fn2 = f'{self.cache_dir}/axial_2d_keypoints_{study_id}_{series_id}_{ins_num}_pts.npy'
        if os.path.exists(fn1):
            img = np.load(fn1)['arr_0']
            keypoints = np.load(fn2)
        else:
            keypoints = sample['keypoints']
            dicom_dir = f'{self.data_dir}/train_images/{study_id}/{series_id}/'
            base_size = 512
            arr, meta = load_dicom(dicom_dir, img_size=base_size)
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

        return {
            'imgs': img,
            'keypoints': keypoints.reshape(-1),
            'mask': mask.reshape(-1)
        }


class Axial_Level_Dataset_Multi(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None,
                 with_origin_arr=False, phase='train'):
        super(Axial_Level_Dataset_Multi, self).__init__()
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
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
        self.samples = []
        if phase in ['train', 'valid']:
            self.df = df[~df['study_id'].isin(hard_axial_study_id_list)]
        else:
            self.df = df
        study_ids = []
        counts = {}
        for _, row in df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            # if len(series_id_list) != 2:
            #     continue
            nn = len(series_id_list)
            if nn == 0:
                print(f'[WARN] {study_id} has no Axial T2')
            if nn not in counts.keys():
                counts[nn] = 1

            counts[nn] += 1
            # study_ids.append(study_id)
            for sid in series_id_list:
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == sid]
                if len(coord_sub_df) > 0:
                    study_ids.append(study_id)
                else:
                    print(f'[WARN] {study_id} has no Axial coord')

        self.df = self.df[self.df['study_id'].isin(study_ids)]
        print(counts)
        self.with_origin_arr = with_origin_arr
        # exit(0)

    def __len__(self):
        return len(self.df)

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

        keypoints = keypoints.reshape(-1, 3)
        return keypoints

    def get_one_series(self, study_id, series_id,
                       base_size=512,
                       crop_size=384,
                       img_size=224):

        keypoints = self.get_axial_coord(study_id, series_id)
        dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
        arr, meta = load_dicom(dicom_dir, img_size=base_size)
        keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=base_size)
        if self.with_origin_arr:
            arr_origin = arr.copy()

        if crop_size != base_size:
            arr = arr.transpose(1, 2, 0)
            arr = A.center_crop(arr, crop_size, crop_size)
            arr = arr.transpose(2, 0, 1)
            x0 = (base_size - crop_size) // 2
            y0 = (base_size - crop_size) // 2
            for i in range(len(keypoints)):
                x, y, ins_num = keypoints[i]
                # no labeled
                if x < 0:
                    continue
                keypoints[i][0] -= x0
                keypoints[i][1] -= y0

        if img_size != crop_size:
            scale = img_size / crop_size
            for i in range(len(keypoints)):
                x, y, ins_num = keypoints[i]
                # no labeled
                if x < 0:
                    continue
                keypoints[i][0] *= scale
                keypoints[i][1] *= scale
        # print('rescaled: ', keypoints)
        # print('PatientPosition: ', meta['PatientPosition'])
        # print('arr shape: ', arr.shape)
        # resample by SpacingBetweenSlices
        SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]

        target_spacing = 3.0
        IPP_z = meta['ImagePositionPatient'][:, 2]
        if target_spacing is not None:
            z_scale = SpacingBetweenSlices / target_spacing
            depth = int(arr.shape[0] * z_scale)

            v = torch.from_numpy(IPP_z).unsqueeze(0).unsqueeze(0)

            v = F.interpolate(v, size=(depth), mode='linear', align_corners=False)
            IPP_z = v.squeeze().numpy()

            # print(IPP_xyz[:, 2].shape)
            # exit(0)

            assert IPP_z.shape[0] == depth

            # target_size = (depth, img_size, img_size)
            target_size = (depth, img_size, img_size)
            arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                size=target_size).numpy().squeeze()

            # keypoints[:, 2] = keypoints[:, 2] * z_scale
            for i in range(keypoints.shape[0]):
                if keypoints[i, 2] != -1:
                    keypoints[i, 2] = keypoints[i, 2] * z_scale

        Z_MIN = -794.237000
        Z_MAX = 351.35
        IPP_z = (IPP_z - Z_MIN) / (Z_MAX - Z_MIN)

        keypoints = keypoints.reshape(5, 2, 3)

        PatientPosition = meta['PatientPosition']
        # if PatientPosition == "FFS":
        #     #print('FFS: ', study_id)
        #     arr = np.flip(arr, -1)
        #     arr = np.ascontiguousarray(arr)
        #     # for i in range(5):
        #     #     if keypoints[i, 0, 0] > 0:
        #     #         keypoints[i, 0, 0] = img_size - keypoints[i, 0, 0]
        #     #     if keypoints[i, 1, 0] > 0:
        #     #         keypoints[i, 1, 0] = img_size - keypoints[i, 1, 0]

        # for i in range(5):
        #     # check
        #     if keypoints[i, 0, 0] > 0 and keypoints[i, 1, 0] > 0:
        #         if keypoints[i, 0, 0] < keypoints[i, 1, 0]:
        #             print('[err]: ', keypoints[:, :, 0])
        #             print(study_id, series_id)
        #             print(PatientPosition)
        #             exit(0)

        left_keypoints = keypoints[:, 0]
        right_keypoints = keypoints[:, 1]
        z_len = arr.shape[0]
        # mean_margin_left = _get_keypoint_mean_margin(left_keypoints)
        # mean_margin_right = _get_keypoint_mean_margin(right_keypoints)
        #
        # z_len = arr.shape[0]
        # if mean_margin_left != -1 and mean_margin_right != -1:
        #     mean_margin = (mean_margin_right + mean_margin_left) / 2
        # elif mean_margin_left == -1 and mean_margin_right != -1:
        #     mean_margin = mean_margin_right
        # elif mean_margin_right == -1 and mean_margin_left != -1:
        #     mean_margin = mean_margin_left
        # else:
        #     mean_margin = 9 * target_spacing / 4.0
        mean_margin = 2
        label, sparse_label_left = _gen_label_by_keypoints(left_keypoints,
                                                           mean_margin, z_len)
        label, sparse_label_right = _gen_label_by_keypoints(right_keypoints,
                                                            mean_margin, z_len, label)

        keypoints = np.concatenate((left_keypoints[:, :2], right_keypoints[:, :2]), axis=0)
        keypoints_z = np.concatenate((left_keypoints[:, 2:], right_keypoints[:, 2:]), axis=0)
        mask = np.where(keypoints < 0, 0, 1)
        mask_z = np.where(keypoints_z < 0, 0, 1)

        if self.transform is not None:
            arr = arr.transpose(1, 2, 0)
            # augmented = self.transform(image=arr, keypoints=keypoints)
            # arr = augmented['image']
            # keypoints = augmented['keypoints']
            # keypoints = np.array(keypoints)

            augmented = self.transform(image=arr)
            arr = augmented['image']
            arr = arr.transpose(2, 0, 1)

        keypoints = np.concatenate((keypoints, keypoints_z), axis=1).reshape(2, 5, 3)
        keypoints_mask = np.concatenate((mask, mask_z), axis=1).reshape(2, 5, 3)

        err = self.check_sparse_label(sparse_label_left, keypoints[0, :, :])
        # print('err: ', err)
        # assert err < 1.0, print(study_id)
        if err > 2.0:
            print(study_id, err)
            exit(0)
        err = self.check_sparse_label(sparse_label_right, keypoints[1, :, :])
        # print('err: ', err)
        # assert err < 1.0, print(study_id)
        if err > 2.0:
            print(study_id, err)
            exit(0)

        sparse_label_left = sparse_label_left[:, np.newaxis, :]
        sparse_label_right = sparse_label_right[:, np.newaxis, :]
        sparse_label = np.concatenate([sparse_label_left, sparse_label_right], axis=1)

        dense_xy, dense_xy_mask, dense_z_mask = \
            self.gen_dense_xy_keypoint_and_mask(arr, keypoints)

        ret = {"arr": arr,
               "label": label,
               "sparse_label": sparse_label,
               "keypoints": keypoints,
               # "keypoints_mask": keypoints_mask,
               "IPP_z": IPP_z,
               "IPP_z0": IPP_z[0],
               "dense_xy": dense_xy,
               "dense_xy_mask": dense_xy_mask,
               "dense_z_mask": dense_z_mask,
               }
        if self.with_origin_arr:
            ret['arr_origin'] = arr_origin
        return ret

    def check_sparse_label(self, sparse_label, pts):
        """

        :param sparse_label: [z_len, 5]
        :param pts:  [5, 3]
        :return:
        """
        z_len, _ = sparse_label.shape
        zs = np.arange(0, z_len)
        errs = []
        for i in range(5):
            z_gt = pts[i, 2]
            if z_gt > 0:
                recover_z = (sparse_label[:, i] * zs).sum() / sparse_label[:, i].sum()
                # print('recover_z: ', recover_z)
                # print('z_gt: ', z_gt)
                errs.append(abs(recover_z - z_gt))
        if len(errs) == 0:
            return 0
        return np.mean(errs)

    def gen_dense_xy_keypoint_and_mask(self,
                                       arr,
                                       keypoints):

        dense_xy = - np.ones((arr.shape[0], 2, 2))
        dense_xy_mask = np.zeros((arr.shape[0], 2, 2))
        dense_z_mask = np.zeros((arr.shape[0], 2, 5))
        for level in range(5):
            # left
            z = keypoints[0, level, 2]
            if z > 0:
                z = int(np.round(z))
                if z > arr.shape[0] - 1:
                    z = arr.shape[0] - 1
                dense_xy[z, 0, 0] = keypoints[0, level, 0]
                dense_xy[z, 0, 1] = keypoints[0, level, 1]
                dense_xy_mask[z, 0, :] = 1
                dense_z_mask[:, 0, level] = 1

            # right
            z = keypoints[1, level, 2]
            if z > 0:
                z = int(np.round(z))
                if z > arr.shape[0] - 1:
                    z = arr.shape[0] - 1
                dense_xy[z, 1, 0] = keypoints[1, level, 0]
                dense_xy[z, 1, 1] = keypoints[1, level, 1]
                dense_xy_mask[z, 1, :] = 1
                dense_z_mask[:, 1, level] = 1
        return dense_xy, dense_xy_mask, dense_z_mask
        # dense_xy_keypoints = []
        # dense_xy_masks = []
        # totol_points = 0
        # for i, arr in enumerate(arr_list):
        #     dense_xy_keypoints.append(torch.from_numpy(dense_xy).float())
        #     dense_xy_masks.append(torch.from_numpy(dense_xy_mask).float())
        #
        # return dense_xy_keypoints, dense_xy_masks

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row['study_id']
        # label = row[1:].values.astype(np.int64)
        # label = label[15:].reshape(2, 5)
        g = self.desc_df[self.desc_df['study_id'] == study_id]
        # print(g)
        series_id_list = g['series_id'].to_list()
        data_list = []

        for sid in series_id_list:
            data_d = self.get_one_series(study_id, sid)
            data_d['sid'] = sid
            data_list.append(data_d)

            # arr: depth, h, w
            # label: depth
            # sparse_label: 2, depth, 5
            # keypoints: 2, 5, 3
            # keypoints_mask:  2, 5, 3
            # IPP_xyz: depth, 1
        data_list = sorted(data_list, key=lambda x: x['IPP_z0'], reverse=True)
        series_id_list = [d['sid'] for d in data_list]
        data_dict = {}
        keys = ['arr', 'label', 'sparse_label',
                'dense_xy', 'dense_xy_mask', 'dense_z_mask',
                'IPP_z']
        for key in keys:
            v_list = []
            for i in range(len(data_list)):
                # print(key, i, data_list[i][key].shape)
                v_list.append(data_list[i][key])
            v = np.concatenate(v_list, axis=0)
            # print(f'{key}: ', v.shape)
            if key in ['label']:
                data_dict[key] = torch.from_numpy(v).long()
            else:
                data_dict[key] = torch.from_numpy(v).float()
        data_dict["study_id"] = study_id

        series_id_dense = []
        index_info = []
        start = 0
        keypoints = []
        coords = []
        for i in range(len(data_list)):
            z_len = data_list[i]['arr'].shape[0]
            index_info.append((start, start + z_len))
            start = z_len
            coords.append(np.arange(0, z_len))
            series_id_dense.append(np.array([i] * z_len))
            keypoints.append(data_list[i]['keypoints'][np.newaxis, :, :, :])
        series_id_dense = np.concatenate(series_id_dense, axis=0)
        data_dict['series_id_dense'] = torch.from_numpy(series_id_dense).long()

        keypoints = np.concatenate(keypoints, axis=0)
        keypoints_mask = np.where(keypoints < 0, 0, 1)
        coords = np.concatenate(coords, axis=0)
        data_dict['xyz_keypoints'] = torch.from_numpy(keypoints).float()
        data_dict['xyz_keypoints_mask'] = torch.from_numpy(keypoints_mask).float()
        data_dict['index_info'] = index_info
        data_dict['coords'] = torch.from_numpy(coords).float()
        data_dict['series_id_list'] = series_id_list
        if self.with_origin_arr:
            arr_origin_list = []
            for i in range(len(data_list)):
                arr = data_list[i]['arr_origin']
                arr = arr.transpose(1, 2, 0)
                augmented = self.transform(image=arr)
                arr = augmented['image']
                arr = arr.transpose(2, 0, 1)
                arr_origin_list.append(torch.from_numpy(arr).float())
            data_dict['arr_origin_list'] = arr_origin_list
        return data_dict


class Axial_Cond_Dataset_Multi(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None,
                 z_imgs=3, phase='test', with_sag=False, sag_transform=None,
                 sag_with_3d=True):
        super(Axial_Cond_Dataset_Multi, self).__init__()
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.data_dir = data_dir
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
        self.z_imgs = z_imgs
        self.with_3_channel = True
        if self.z_imgs == 3:
            self.with_3_channel = False

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
        self.samples = []

        if phase == 'train':
            self.df = df[~df['study_id'].isin(hard_axial_study_id_list)]
        else:
            self.df = df
        self.df = df[~df['study_id'].isin(hard_axial_study_id_list)]

        study_ids = []
        counts = {}
        for _, row in df.iterrows():
            study_id = row['study_id']
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            # if len(series_id_list) != 1:
            #     continue
            nn = len(series_id_list)
            if nn not in counts.keys():
                counts[nn] = 1

            counts[nn] += 1
            # study_ids.append(study_id)
            for sid in series_id_list:
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == sid]
                if len(coord_sub_df) > 0:
                    study_ids.append(study_id)

        self.df = self.df[self.df['study_id'].isin(study_ids)]
        print('samples: ', len(self.df))
        print('z_imgs: ', z_imgs)
        self.cache_dir = self.data_dir + '/cache/'
        create_dir(self.cache_dir)
        # self.axial_pred_keypoints_info = pickle.load(
        #     open(f'{data_dir}/v24_axial_pred_en5.pkl', 'rb'))


        # self.axial_pred_keypoints_info = pickle.load(
        #     open(f'{data_dir}/v24_axial_pred_fold1_pvt.pkl', 'rb'))
        #

        # self.axial_pred_keypoints_info = pickle.load(
        #     open(f'{data_dir}/v24_axial_pred_en5_no_ipp2.pkl', 'rb'))
        self.axial_pred_keypoints_info = pickle.load(
            open(f'{data_dir}/v24_axial_pred_fold0_no_ipp2.pkl', 'rb'))

        self.phase = phase
        self.with_sag = with_sag
        self.sag_with_3d = sag_with_3d
        print('with_sag: ', with_sag)
        if with_sag:
            self.sag_t2_dset = Sag_T2_Dataset(data_dir,
                                              df, transform=sag_transform,
                                              phase=phase,
                                              z_imgs=None if sag_with_3d else 7,
                                              img_size=512,
                                              crop_size_h=64,
                                              crop_size_w=128,
                                              resize_to_size=128
                                              )

    def __len__(self):
        return len(self.df)

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

        keypoints = keypoints.reshape(-1, 3)
        return keypoints

    def load_cache_one(self, study_id, series_id,
                       base_size=512,
                       crop_size=256,
                       img_size=256,
                       target_spacing=3.0):
        fn1 = f'{self.cache_dir}/{study_id}_{series_id}_arr.npz'
        fn2 = f'{self.cache_dir}/{study_id}_{series_id}_pts.npy'
        if False:#os.path.exists(fn1):
            arr = np.load(fn1)['arr_0']
            keypoints = np.load(fn2)
            return arr, keypoints
        else:
            arr, keypoints = self.get_one_series(study_id,
                                                 series_id, base_size,
                                                 crop_size, img_size, target_spacing)
            np.savez_compressed(fn1, arr)
            np.save(fn2, keypoints)
            return arr, keypoints

    def get_one_series(self, study_id, series_id,
                       base_size=512,
                       crop_size=256,
                       img_size=256,
                       target_spacing=3.0):

        keypoints = self.get_axial_coord(study_id, series_id)
        dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
        arr, meta = load_dicom(dicom_dir, img_size=base_size)

        # PatientPosition = meta['PatientPosition']
        # if PatientPosition == "FFS":
        #     arr = np.flip(arr, -1)
        #     arr = np.ascontiguousarray(arr)

        keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=base_size)

        if crop_size != base_size:
            arr = arr.transpose(1, 2, 0)
            arr = A.center_crop(arr, crop_size, crop_size)
            arr = arr.transpose(2, 0, 1)
            x0 = (base_size - crop_size) // 2
            y0 = (base_size - crop_size) // 2
            for i in range(len(keypoints)):
                x, y, ins_num = keypoints[i]
                # no labeled
                if x < 0:
                    continue
                keypoints[i][0] -= x0
                keypoints[i][1] -= y0
        # assert img_size == crop_size, 'TODO'
        if img_size != crop_size:
            scale = img_size / crop_size
            for i in range(len(keypoints)):
                x, y, ins_num = keypoints[i]
                # no labeled
                if x < 0:
                    continue
                keypoints[i][0] *= scale
                keypoints[i][1] *= scale
        # keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=img_size)
        if target_spacing != None:
            SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]
            z_scale = SpacingBetweenSlices / target_spacing
            depth = int(arr.shape[0] * z_scale)
            target_size = (depth, img_size, img_size)
            arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                size=target_size).numpy().squeeze()
            for i in range(keypoints.shape[0]):
                if keypoints[i, 2] != -1:
                    keypoints[i, 2] = keypoints[i, 2] * z_scale

        keypoints = keypoints.reshape(5, 2, 3)
        return arr, keypoints

    def crop_by_z(self, volume, z, z_imgs=3):
        z0 = z - z_imgs // 2
        z1 = z + z_imgs // 2 + 1
        if z0 < 0:
            z0 = 0
            z1 = z_imgs
        if z1 > volume.shape[0]:
            z0 = volume.shape[0] - z_imgs
            z1 = volume.shape[0]
        # print(z0, z1)
        v = volume[z0: z1, ].copy()
        return v

    def assemble_5_level_imgs_with_transform(self,
                                             level_to_imgs_dict,
                                             z_imgs=3, image_size=256,
                                             transform=None):
        imgs = []
        for level in range(5):
            if level not in level_to_imgs_dict.keys():
                if self.with_3_channel:
                    img = np.zeros((z_imgs, 3, image_size, image_size), dtype=np.float32)
                else:
                    img = np.zeros((z_imgs, image_size, image_size), dtype=np.float32)
                imgs.append(img)
            else:
                if len(level_to_imgs_dict[level]) > 1:
                    level_to_imgs_dict[level] = sorted(level_to_imgs_dict[level],
                                                       key=lambda x: x[1], reverse=True)
                    # for img, score in level_to_imgs_dict[level]:
                    #     print(img.shape)
                    #     print(score)
                    # exit(0)
                    # print('[WARN todo] len level_to_imgs_dict[level]: ', len(level_to_imgs_dict[level]))
                img, score = level_to_imgs_dict[level][0]
                # print(img.shape, score)
                # exit(0)
                if self.with_3_channel:
                    ind = np.arange(img.shape[0])
                    ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
                    ind_a = np.where(np.array(ind) + 1 >= img.shape[0] - 1, img.shape[0] - 1, np.array(ind) + 1)
                    img = np.stack([img[ind_b, :, :], img, img[ind_a, :, :]], axis=-1)

                if transform is not None:
                    img = [transform(image=i)['image'] for i in img]
                    img = np.array(img)
                    if self.with_3_channel:
                        img = img.transpose(0, 3, 1, 2)
                    # img = img.transpose(1, 2, 0)
                    # img = transform(image=img)['image']
                    # img = img.transpose(2, 0, 1)
                if not self.with_3_channel:
                    img = img[:, np.newaxis, :, :]
                imgs.append(img)
        imgs = np.array(imgs)  # 5, 5, 3, 256, 256

        # print('imgs shape: ', imgs.shape)
        # exit(0)
        imgs = np.array(imgs, dtype=np.float32)
        return imgs

    def assemble_5_level_imgs_with_transform_v2(self,
                                                level_to_imgs_dict,
                                                z_imgs=3, image_size=256,
                                                transform=None):
        imgs = []
        levels = []
        for level in range(5):
            if level not in level_to_imgs_dict.keys():
                img = np.zeros((z_imgs, 3, image_size, image_size), dtype=np.float32)
                # img = np.zeros((z_imgs, image_size, image_size), dtype=np.float32)
                imgs.append(img)
            else:
                if len(level_to_imgs_dict[level]) > 1:
                    level_to_imgs_dict[level] = sorted(level_to_imgs_dict[level],
                                                       key=lambda x: x[1], reverse=True)
                    # for img, score in level_to_imgs_dict[level]:
                    #     print(img.shape)
                    #     print(score)
                    # exit(0)
                    # print('[WARN todo] len level_to_imgs_dict[level]: ', len(level_to_imgs_dict[level]))
                img, score = level_to_imgs_dict[level][0]
                # print(img.shape, score)
                # exit(0)
                ind = np.arange(img.shape[0])
                ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
                ind_a = np.where(np.array(ind) + 1 >= img.shape[0] - 1, img.shape[0] - 1, np.array(ind) + 1)
                img = np.stack([img[ind_b, :, :], img, img[ind_a, :, :]], axis=-1)

                if transform is not None:
                    img = [transform(image=i)['image'] for i in img]
                    img = np.array(img)
                    img = img.transpose(0, 3, 1, 2)
                    # img = img.transpose(1, 2, 0)
                    # img = transform(image=img)['image']
                    # img = img.transpose(2, 0, 1)
                imgs.append(img)
        imgs = np.array(imgs)  # 5, 5, 3, 256, 256

        # print('imgs shape: ', imgs.shape)
        # exit(0)
        imgs = np.array(imgs, dtype=np.float32)
        return imgs

    def crop_out_by_keypoints(self, volume, keypoints_xyz,
                              z_imgs=3,
                              crop_size_h=128,
                              crop_size_w=128,
                              transform=None,
                              resize_to_size=None):
        G_IMG_SIZE = 512
        sub_volume_list = []
        for p in keypoints_xyz:
            x, y, z = np.round(p[0]), np.round(p[1]), np.round(p[2])
            x, y, z = int(x), int(y), int(z)
            # no z
            if z < 0:
                v = np.zeros((z_imgs, crop_size_h, crop_size_w), dtype=volume.dtype)
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

        try:
            volume_crop = np.array(sub_volume_list)
        except:
            print(volume.shape)
            print(crop_size_w, crop_size_h)
            for s in sub_volume_list:
                print(s.shape)
            for p in keypoints_xyz:
                x, y, z = np.round(p[0]), np.round(p[1]), np.round(p[2])
                x, y, z = int(x), int(y), int(z)
                bbox = [x - crop_size_w // 2,
                        y - crop_size_h // 2,
                        x + crop_size_w // 2,
                        y + crop_size_h // 2]
                print('b0: ', bbox, z)
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
                print('b1: ', bbox)
            exit(0)

        if resize_to_size is not None:
            volume_crop = torch.from_numpy(volume_crop)
            volume_crop = F.interpolate(volume_crop, (resize_to_size, resize_to_size))
            volume_crop = volume_crop.numpy()

        return volume_crop, None

    def get_pred_keypoints(self, study_id, sid, arr,
                           base_size=720, crop_size=512, img_size=512, ):
        pts = self.axial_pred_keypoints_info[study_id][sid]['points']  # 2, 5, 4
        pts = pts.transpose(1, 0, 2)  # 5, 2, 4
        pts = np.ascontiguousarray(pts)
        # print('pts: ',pts.shape)
        z_len, h, w = arr.shape

        origin_base_size = 512
        origin_crop_size = 384
        origin_img_size = 224
        # x0 = origin_base_size // 2 - origin_crop_size // 2
        # y0 = origin_base_size // 2 - origin_crop_size // 2
        # x = pts[:, :, 0] * origin_crop_size + x0
        # y = pts[:, :, 1] * origin_crop_size + y0
        # print('x: ', x)
        x = pts[:, :, 0] * origin_base_size
        y = pts[:, :, 1] * origin_base_size

        scale = base_size / origin_base_size
        # print('scale: ', scale)
        x *= scale
        y *= scale
        # print('x: ', x)

        x0 = (base_size - crop_size) // 2
        y0 = (base_size - crop_size) // 2
        x = x - x0
        y = y - y0
        # x = x * img_size / crop_size
        # y = y * img_size / crop_size
        pts[:, :, 0] = x
        pts[:, :, 1] = y

        pts[:, :, 2] = pts[:, :, 2] * z_len
        return pts

    def avg_left_right_pts(self, pts):
        keypoints_mean = -np.ones((5, 4), dtype=np.float32)
        for i in range(5):
            x0, y0, z0, s0 = pts[i][0]
            x1, y1, z1, s1 = pts[i][1]
            if z0 > 0 and z1 > 0:
                keypoints_mean[i, 0] = (x0 + x1) / 2
                keypoints_mean[i, 1] = (y0 + y1) / 2
                keypoints_mean[i, 2] = (z0 + z1) / 2
                keypoints_mean[i, 3] = (s0 + s1) / 2
            elif z0 > 0:
                keypoints_mean[i, 0] = x0
                keypoints_mean[i, 1] = y0
                keypoints_mean[i, 2] = z0
                keypoints_mean[i, 3] = s0
            elif z1 > 0:
                keypoints_mean[i, 0] = x1
                keypoints_mean[i, 1] = y1
                keypoints_mean[i, 2] = z1
                keypoints_mean[i, 3] = s1
            else:
                pass
        return keypoints_mean

    # def get_item_v2(self, idx):
    #     row = self.df.iloc[idx]
    #     study_id = row['study_id']
    #     label = row[1:].values.astype(np.int64)
    #     label = label[15:].reshape(2, 5)
    #
    #     g = self.desc_df[self.desc_df['study_id'] == study_id]
    #     series_id_list = g['series_id'].to_list()
    #     arr_list = []
    #     keypoints_list = []
    #     base_size = 720
    #     crop_size = 512
    #     img_size = 512
    #     target_spacing = None
    #
    #     for sid in series_id_list:
    #         arr, gt_keypoints = self.load_cache_one(study_id, sid,
    #                                                 base_size, crop_size,
    #                                                 img_size, target_spacing)
    #         scores = np.zeros((5, 2, 1), dtype=np.float32)
    #         gt_keypoints = np.concatenate((gt_keypoints, scores), axis=-1)
    #
    #         arr_list.append(arr)
    #         pred_keypoints = self.get_pred_keypoints(study_id, sid, arr,
    #                                                  base_size, crop_size, img_size, )
    #
    #         z_mask = np.where(gt_keypoints[:, :, 2] < 0, 0, 1)
    #         z_err = np.mean(np.abs(pred_keypoints[:, :, 2] * z_mask -
    #                                gt_keypoints[:, :, 2] * z_mask))
    #         use_gt_z = False
    #         if z_err > 2:
    #             # print('err: ', z_err, study_id)
    #             use_gt_z = True
    #         if use_gt_z:
    #             pred_keypoints[:, :, 2] = gt_keypoints[:, :, 2]
    #
    #         # 5, 4
    #         #gt_keypoints_mean = self.avg_left_right_pts(gt_keypoints)
    #         #print('pred_keypoints shape: ', pred_keypoints.shape)
    #         pred_keypoints_mean = self.avg_left_right_pts(pred_keypoints)
    #
    #         keypoints_list.append(pred_keypoints_mean)
    #
    #     level_to_imgs = {}
    #     # debug_imgs = []
    #     roi_img_size = 224
    #     for i in range(len(keypoints_list)):
    #         keypoints = keypoints_list[i]
    #         arr = arr_list[i]
    #         for level in range(5):
    #             p = keypoints[level]
    #             if p[2] > 0:
    #                 img, _ = self.crop_out_by_keypoints(arr,
    #                                                     [p],
    #                                                     z_imgs=self.z_imgs,
    #                                                     crop_size_w=roi_img_size,
    #                                                     crop_size_h=roi_img_size)
    #                 img = img[0]
    #                 if level in level_to_imgs.keys():
    #                     level_to_imgs[level].append([img, p[3]])
    #                 else:
    #                     level_to_imgs[level] = [[img, p[3]]]
    #
    #     imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs,
    #                                                           z_imgs=self.z_imgs,
    #                                                           image_size=roi_img_size,
    #                                                           transform=self.transform)
    #     axial_imgs = imgs.astype(np.float32)
    #     data_dict = {
    #         'axial_imgs': axial_imgs,
    #         'label': label.reshape(-1)
    #     }
    #
    #     return data_dict

    def __getitem__(self, idx):
        # return self.get_item_v2(idx)

        row = self.df.iloc[idx]
        study_id = row['study_id']
        label = row[1:].values.astype(np.int64)
        label = label[15:].reshape(2, 5)

        g = self.desc_df[self.desc_df['study_id'] == study_id]
        series_id_list = g['series_id'].to_list()
        arr_list = []
        keypoints_list = []
        base_size = 720
        crop_size = 512
        img_size = 512
        target_spacing = None

        for sid in series_id_list:
            arr, gt_keypoints = self.load_cache_one(study_id, sid,
                                                    base_size, crop_size,
                                                    img_size, target_spacing)
            scores = np.ones((5, 2, 1), dtype=np.float32)
            gt_keypoints = np.concatenate((gt_keypoints, scores), axis=-1)
            arr_list.append(arr)
            if True:  # len(series_id_list) == 1:
                pred_keypoints = self.get_pred_keypoints(study_id, sid, arr,
                                                         base_size, crop_size, img_size, )

                # pred_keypoints[:, :, :2] = gt_keypoints[:, :, :2]
                # mask = np.where(gt_keypoints[:, :, :2]<0, 0, 1)
                # err = np.mean(np.abs(pred_keypoints[:, :, :2] * mask -
                #                      gt_keypoints[:, :, :2] * mask))
                # print('err: ', err)
                # if err > 10:
                #     print('err: ', err, study_id)
                pred_keypoints = np.round(pred_keypoints)
                z_mask = np.where(gt_keypoints[:, :, 2] < 0, 0, 1)
                z_err = np.mean(np.abs(pred_keypoints[:, :, 2] * z_mask -
                                       gt_keypoints[:, :, 2] * z_mask))
                # print(pred_keypoints[:, :, 2])
                # print(gt_keypoints[:, :, 2])
                # print('z_err: ',z_err)
                # use_gt_z = False
                if z_err > 2 and self.phase == 'train':
                    pred_keypoints = gt_keypoints

                # pred_keypoints = gt_keypoints
                # pred_keypoints[:, :, :2] = gt_keypoints[:, :, :2]
                if z_err > 0.5 and self.phase == 'train':
                    pred_keypoints[:, :, 2] = gt_keypoints[:, :, 2]

                if self.phase == 'train' and random.random() < 0.5:
                    if random.random() < 0.5:
                        gt_keypoints[:, :, :2] = pred_keypoints[:, :, :2]
                    keypoints_list.append(gt_keypoints)
                else:
                    keypoints_list.append(pred_keypoints)

            else:
                keypoints_list.append(gt_keypoints)

        level_to_imgs_left = {}
        level_to_imgs_right = {}
        # debug_imgs = []
        roi_img_size = 128
        for i in range(len(keypoints_list)):
            keypoints = keypoints_list[i]
            arr = arr_list[i]
            for level in range(5):
                left_p = keypoints[level, 0]
                if left_p[2] > 0:
                    z = int(np.round(left_p[2]))
                    # img = self.crop_by_z(arr, z, z_imgs=self.z_imgs)

                    img, _ = self.crop_out_by_keypoints(arr,
                                                        [left_p],
                                                        z_imgs=self.z_imgs,
                                                        crop_size_w=roi_img_size,
                                                        crop_size_h=roi_img_size)
                    img = img[0]

                    if level in level_to_imgs_left.keys():
                        level_to_imgs_left[level].append([img, left_p[3]])
                    else:
                        level_to_imgs_left[level] = [[img, left_p[3]]]

                    # x,y,z = left_p
                    # x, y, z = int(x), int(y), int(z)
                    # print(f'level: {level}, xyz: ', x, y,z, i)
                    # img  = img.transpose(1, 2, 0)
                    # img  = np.ascontiguousarray(img)
                    #
                    # #img = 255 * (img - img.min()) / (img.max() - img.min())
                    # img = np.array(img, dtype=np.uint8)
                    #
                    # img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
                    # cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.putText(img, 'level: ' + str(level), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    #             cv2.LINE_AA)
                    # sid = i
                    # cv2.putText(img, 'sid: ' + str(sid), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    #             cv2.LINE_AA)
                    # debug_imgs.append(img)

                right_p = keypoints[level, 1]
                if right_p[2] > 0:
                    z = int(np.round(right_p[2]))
                    # img = self.crop_by_z(arr, z, z_imgs=self.z_imgs)
                    img, _ = self.crop_out_by_keypoints(arr,
                                                        [right_p],
                                                        z_imgs=self.z_imgs,
                                                        crop_size_w=roi_img_size,
                                                        crop_size_h=roi_img_size)
                    img = img[0]

                    if level in level_to_imgs_right.keys():
                        level_to_imgs_right[level].append([img, right_p[3]])
                    else:
                        level_to_imgs_right[level] = [[img, right_p[3]]]
        #
        # img_concat = np.concatenate(debug_imgs, axis=0)
        # cv2.imwrite(f'{self.data_dir}/debug_dir/0_v24_axial_cond.jpg', img_concat)
        #
        for level in range(5):
            if level not in level_to_imgs_left.keys():
                label[0, level] = -100
            if level not in level_to_imgs_right.keys():
                label[1, level] = -100
        left_imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs_left,
                                                              z_imgs=self.z_imgs,
                                                              image_size=roi_img_size,
                                                              transform=self.transform)
        right_imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs_right,
                                                               z_imgs=self.z_imgs,
                                                               image_size=roi_img_size,
                                                               transform=self.transform)
        right_imgs = np.flip(right_imgs, axis=[-1])
        axial_imgs = np.concatenate((left_imgs, right_imgs), axis=0)
        axial_imgs = axial_imgs.astype(np.float32)

        data_dict = {
            'axial_imgs': axial_imgs,
            'label': label.reshape(-1),
            'study_id': study_id
        }
        if self.with_sag:
            sag_t2 = self.sag_t2_dset.get_item_by_study_id(study_id)['s_t2']
            # print('sag_t2: ', sag_t2.shape)
            if self.sag_with_3d:
                sag_t2 = torch.from_numpy(sag_t2).unsqueeze(1)
                sag_t2 = F.interpolate(sag_t2, size=(32, 96, 96)).squeeze().numpy()
            else:
                # sag_t2 = torch.from_numpy(sag_t2).unsqueeze(1)
                # sag_t2 = F.interpolate(sag_t2, size=(16, 128, 128)).squeeze().numpy()
                # sag_t2 = sag_t2[:, 3:13]

                cs = sag_t2.shape[1] // 2
                left_side = sag_t2[:, cs:, :, :]
                # right_side = sag_t2[:, :cs, :, :]
                right_side = sag_t2[:, :cs + 1, :, :]
                right_side = np.flip(right_side, axis=[1])
                right_side = np.ascontiguousarray(right_side)
                sag_t2 = np.concatenate((left_side, right_side), axis=0)

            data_dict['sag_t2'] = sag_t2.astype(np.float32)

        return data_dict


def collate_fn(batch):
    return batch
    # ret = {
    #     'arr_list_list': [],
    #     'label_list_list': [],
    #     'sparse_label_list_list': [],
    #     'keypoints_list_list': [],
    #     'keypoints_mask_list_list': [],
    #     'IPP_xyz_list_list': [],
    #     'study_id_list': [],
    #     'dense_xy_keypoints': [],
    #     'dense_xy_mask': []
    # }
    # for i in range(len(batch)):
    #     ret['arr_list_list'].append(batch[i]['arr_list'])
    #     ret['label_list_list'].append(batch[i]['label_list'])
    #     ret['sparse_label_list_list'].append(batch[i]['sparse_label_list'])
    #     ret['keypoints_list_list'].append(batch[i]['keypoints_list'])
    #     ret['keypoints_mask_list_list'].append(batch[i]['keypoints_mask_list'])
    #     ret['IPP_xyz_list_list'].append(batch[i]['IPP_xyz_list'])
    #     ret['study_id_list'].append(batch[i]['study_id'])
    #     ret['dense_xy_keypoints'].append(batch[i]['dense_xy_keypoints'])
    #     ret['dense_xy_mask'].append(batch[i]['dense_xy_mask'])
    # return ret


def data_to_cuda(d):
    bs = len(d)
    for b in range(bs):
        for k in d[b].keys():
            if k not in ['study_id', 'index_info', 'series_id_list']:
                if isinstance(d[b][k], list):
                    for n in range(len(d[b][k])):
                        d[b][k][n] = d[b][k][n].cuda()
                else:
                    d[b][k] = d[b][k].cuda()
    return d


# def data_to_cuda(d):
#     for k in d.keys():
#         if k == 'study_id_list':
#             continue
#         for i in range(len(d[k])):
#             for j in range(len(d[k][i])):
#                 d[k][i][j] = d[k][i][j].cuda()
#     return d

def convert_to_cv2_img(volume):
    k, d, h, w = volume.shape
    img = np.zeros((k * h, d * w))
    for ik in range(k):
        for id in range(d):
            img[ik * h: h * (ik + 1), id * w: w * (id + 1)] = volume[ik, id]

    img = 255 * (img - img.min()) / (img.max() - img.min())
    return img


def test_cond_dataset():
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # df = df[df['study_id'] == 2022619830]

    import albumentations as A

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    debug_dir = f'{data_root}/debug_dir/'

    dset = Axial_Cond_Dataset_Multi(data_root, df, transform=transforms_train,
                                    z_imgs=5, with_sag=True, sag_with_3d=False)
    dloader = DataLoader(dset, num_workers=0, batch_size=1)
    for d in dloader:
        imgs = d['axial_imgs'][:, :, :, 1]
        print(imgs.shape)
        cv2.imwrite(f'{debug_dir}/0_v24_axial_cond.jpg', convert_to_cv2_img(imgs[0]))

        imgs = d['sag_t2'][:, :, :]
        print(imgs.shape)
        cv2.imwrite(f'{debug_dir}/0_v24_axial_cond_sag.jpg', convert_to_cv2_img(imgs[0]))

        break


def test_dataset():
    import tqdm
    import matplotlib.pyplot as plt
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    df = df[df['study_id'] == 2022619830]

    import albumentations as A
    from train_scripts.v24.axial_model import Axial_Level_Cls_Model

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    debug_dir = f'{data_root}/debug_dir/'

    dset = Axial_Level_Dataset_Multi(data_root, df, transform=transforms_train)
    dloader = DataLoader(dset, num_workers=0, batch_size=1, collate_fn=collate_fn)
    model = Axial_Level_Cls_Model('convnext_small.in12k_ft_in1k_384', pretrained=False).cuda()
    model_dir = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/v24/pretrain_axial_level_cls/convnext_small.in12k_ft_in1k_384'
    model.load_state_dict(
        torch.load(f'{model_dir}/best_fold_0_ema.pt'))
    model.eval()
    autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)

    for d in tqdm.tqdm(dloader):

        arr = d[0]['arr']
        dense_xy = d[0]['dense_xy']
        label = d[0]['label']
        series_id_dense = d[0]['series_id_dense']
        sparse_label = d[0]['sparse_label']  # seq_len, 2, 5
        dense_z_mask = d[0]['dense_z_mask']

        IPP_z = d[0]['IPP_z']

        # 绘制5条曲线
        time_steps = np.arange(sparse_label.shape[0])
        for i in range(5):
            plt.plot(time_steps, sparse_label[:, 0, i], label=f"Curve_left{i + 1}")
            plt.plot(time_steps, dense_z_mask[:, 0, i], label=f"z_mask_left{i}")

        # plt.plot(time_steps, IPP_z, label=f"ipp_z")
        # 设置图例和标题
        plt.legend()
        plt.title('GT left')
        plt.xlabel('instance_num')
        plt.ylabel('Confidence Scores')
        plt.show()

        for i in range(5):
            plt.plot(time_steps, sparse_label[:, 1, i], label=f"Curve_right{i + 1}")
            plt.plot(time_steps, dense_z_mask[:, 1, i], label=f"z_mask_right{i}")
        plt.legend()
        plt.title('GT right')
        plt.xlabel('instance_num')
        plt.ylabel('Confidence Scores')
        plt.show()

        with torch.no_grad():
            # with autocast:
            pred_dict = model(data_to_cuda(d))
            sparse_pred = pred_dict['sparse_pred'].cpu().float().sigmoid().numpy()
        sparse_pred = sparse_pred[0]
        # sparse_pred = sparse_pred.mean(axis=1)
        for i in range(5):
            plt.plot(time_steps, sparse_pred[:, 0, i], label=f"Level_{i + 1}")

        plt.legend()
        plt.title('Pred Left')
        plt.xlabel('instance_num')
        plt.ylabel('Confidence Scores')
        plt.show()

        for i in range(5):
            plt.plot(time_steps, sparse_pred[:, 1, i], label=f"Level_{i + 1}")

        plt.legend()
        plt.title('Pred Right')
        plt.xlabel('instance_num')
        plt.ylabel('Confidence Scores')
        plt.show()

        imgs = []
        for z, p2 in enumerate(dense_xy):
            p = p2[0]  # right
            x, y = int(p[0]), int(p[1])
            if x < 0:
                continue
            img = arr[z]
            print('xyz: ', x, y, z)
            img = 255 * (img - img.min()) / (img.max() - img.min())
            img = np.array(img, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
            cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            la = int(label[z])
            cv2.putText(img, 'level: ' + str(la), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            sid = int(series_id_dense[z])
            cv2.putText(img, 'sid: ' + str(sid), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            imgs.append(img)

        img_concat = np.concatenate(imgs, axis=0)
        cv2.imwrite(f'{debug_dir}/0_v24_axial_data.jpg', img_concat)
        exit(0)

    # depth_list = []
    # z_list = []
    # for d in tqdm.tqdm(dloader):
    #     #print('len: ', len(d['IPP_xyz_list_list']))
    #     IPP_xyz_list = d['IPP_xyz_list_list'][0]
    #     for IPP_xyz in IPP_xyz_list:
    #         #print(IPP_xyz.shape)
    #         z_list.append(np.min(IPP_xyz[:, 2].numpy()))
    #         z_list.append(np.max(IPP_xyz[:, 2].numpy()))
    #
    #     arr_list = d['arr_list_list'][0]
    #     for arr in arr_list:
    #         depth_list.append(arr.shape[0])
    #
    # print(pd.DataFrame(z_list).describe())
    # print(pd.DataFrame(depth_list).describe())


def test_axial_2d_point_dataset():
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    import albumentations as A

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    debug_dir = f'{data_root}/debug_dir/'

    dset = Axial_2D_Point_Dataset(data_root, df, transform=transforms_train)
    dloader = DataLoader(dset, num_workers=0, batch_size=1)
    for d in dloader:
        img = d['imgs'][0][0]
        keypoints = d['keypoints'][0].reshape(2, 2)
        keypoints = np.asarray(keypoints, np.int64)
        mask = d['mask'][0].reshape(2, 2)
        print(keypoints)
        print(mask)
        print(img.shape)
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x, y = keypoints[0]
        img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
        x, y = keypoints[1]
        img = cv2.circle(img, (x, y), 5, (255, 0, 255), 9)
        cv2.imwrite(f'{debug_dir}/0_v24_axial_2d_keypoints.jpg', img)
        break


if __name__ == '__main__':
    # test_dataset()
    # test_cond_dataset()
    # test_axial_2d_point_dataset()
    # exit(0)
    import tqdm

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # df = df[df['study_id']==26342422]
    transforms_val = A.Compose([
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = Axial_Level_Dataset_Multi(data_root, df, transform=transforms_val)
    valid_dl = DataLoader(dset, num_workers=12, batch_size=1)
    model = Axial_Level_Cls_Model("convnext_small.in12k_ft_in1k_384", pretrained=False)
    model.cuda()
    model_dir = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/v24/pretrain_axial_level_cls_no_ipp2/convnext_small.in12k_ft_in1k_384/'
    model.load_state_dict(
        torch.load(f'{model_dir}/best_fold_0_ema.pt'))
    model.eval()
    autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)

    y_preds = []
    labels = []
    with tqdm.tqdm(valid_dl, leave=True) as pbar:
        with torch.no_grad():
            for idx, tensor_dict in enumerate(pbar):

                img = tensor_dict['img'].cuda()
                label = tensor_dict['label'].cuda()

                with autocast:
                    pred = model(img)
                    pred = pred.cpu().reshape(-1, 5)
                    label = label.cpu().reshape(-1)

                    y_preds.append(pred)
                    labels.append(label)

                pred = pred.float().softmax(dim=-1).numpy()  # n, 5
                label = label.numpy()
                mask = np.where(label != -100)
                pred = pred[mask]
                label = label[mask]
                pred_cls = np.argmax(pred, axis=1)
                acc = accuracy_score(label, pred_cls)
                pred_prob = []
                for i in range(len(pred_cls)):
                    pred_prob.append(pred[i, pred_cls[i]])
                # print('acc: ', acc)
                if acc < 0.4:
                    print('acc: ', acc)
                    print(tensor_dict['study_id'][0])
                    print(tensor_dict['series_id'][0])
                    print(img.shape)
                    print('label: ', label)
                    print('pred: ', pred_cls)
                    print('pred prob: ', pred_prob)
                    print('====' * 10)

    labels = torch.cat(labels)  # n,
    y_preds = torch.cat(y_preds, dim=0).float().softmax(dim=-1).numpy()  # n, 5

    mask = np.where(labels != -100)
    y_preds = y_preds[mask]
    labels = labels[mask]

    # calculate acc by sklean
    y_preds = np.argmax(y_preds, axis=1)
    acc = accuracy_score(labels, y_preds)
    print('acc: ', acc)
