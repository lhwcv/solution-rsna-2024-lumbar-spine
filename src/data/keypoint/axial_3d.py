# -*- coding: utf-8 -*-
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
from src.utils.aux_info import get_train_study_aux_info
import torch.nn.functional as F
from skimage.transform import resize

_level_to_idx = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}


class RSNA24Dataset_KeyPoint_Axial_3D(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 img_size=512):

        self.df = df
        self.aux_info = aux_info
        self.transform = transform

        self.phase = phase
        self.volume_data_root = f'{data_root}/train_images_preprocessed/'
        self.img_size = img_size

        self.samples = []
        for i, row in df.iterrows():
            label = row[1:].values.astype(np.int64)
            study_id = row['study_id']
            label_info = self.aux_info[study_id]['aux_infos']
            dataframes = {
                'series_id': [],
                'instance_number': [],
                'x': [],
                'y': [],
                'level': [],
                'cond': [],
            }
            for cond, coord_labels in label_info.items():
                if cond not in ['Left Subarticular Stenosis', 'Right Subarticular Stenosis']:
                    continue
                for a in coord_labels:
                    dataframes['series_id'].append(a['series_id'])
                    dataframes['instance_number'].append(a['instance_number'])
                    dataframes['x'].append(a['x'])
                    dataframes['y'].append(a['y'])
                    dataframes['level'].append(a['level'])
                    dataframes['cond'].append(cond)

            dataframes = pd.DataFrame(dataframes)
            for series_id in dataframes['series_id'].unique():
                sub_dfs = dataframes[dataframes['series_id'] == series_id]
                # convert g to list of dict
                g = sub_dfs.to_dict('records')
                # print(g)
                # exit(0)
                self.samples.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'labeled_keypoints': g,
                    'label': label
                })

        print('[RSNA24Dataset_KeyPoint_Axial_3D] samples: ', len(self.samples))
        # print(self.samples[0])
        # exit(0)
        self.gt_keypoint_info = self.build_gt_keypoints()
        # # print(self.gt_keypoint_info[11943292][3800798510])
        # nums_1_sid = 0
        # nums_has_complete_label = 0
        # labeled_levels = []
        # for study_id in self.gt_keypoint_info.keys():
        #     if len(self.gt_keypoint_info[study_id].keys()) == 1:
        #         nums_1_sid += 1
        #     has_labeled_levels = set()
        #     for series_id, v in self.gt_keypoint_info[study_id].items():
        #         for l in v['has_labeled_levels']:
        #             has_labeled_levels.add(l)
        #         # if len(v['has_labeled_levels']) == 5:
        #         #     nums_has_complete_label += 1
        #         # else:
        #         #     labeled_levels.append(','.join(v['has_labeled_levels']))
        #     if len(has_labeled_levels) == 5:
        #         nums_has_complete_label += 1
        #     else:
        #         has_labeled_levels = sorted(has_labeled_levels)
        #         labeled_levels.append(','.join(has_labeled_levels))
        #
        # print('nums_1_sid: ', nums_1_sid)
        # print('nums_has_complete_label: ', nums_has_complete_label)
        # labeled_levels = pd.DataFrame(labeled_levels, columns=['levels'])
        # print(labeled_levels.value_counts())
        #
        # exit(0)

    def __len__(self):
        return len(self.samples)

    def load_volume(self, fns, index, bbox=None):
        arr = []
        if index is None:
            index = range(len(fns))
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

    def build_gt_keypoints(self):
        gt_keypoint_info = {}
        for item in self.samples:
            study_id = int(item['study_id'])
            series_id = int(item['series_id'])
            meta_fn = f'{self.volume_data_root}/{study_id}/{series_id}/meta_info_2.pkl'
            meta = pickle.load(open(meta_fn, 'rb'))
            instance_num_to_shape = meta['instance_num_to_shape']
            dicom_instance_numbers_to_idx = {}
            for i, ins in enumerate(meta['dicom_instance_numbers']):
                dicom_instance_numbers_to_idx[ins] = i

            keypoints_left = -np.ones((5, 2), dtype=np.float32)
            keypoints_right = -np.ones((5, 2), dtype=np.float32)
            keypoints_z_left = -np.ones((5, 1), dtype=np.float32)
            keypoints_z_right = -np.ones((5, 1), dtype=np.float32)
            instance_numbers = set()
            has_labeled_levels = set()
            for a in item['labeled_keypoints']:
                instance_number = a['instance_number']
                instance_numbers.add(instance_number)
                x, y = a['x'], a['y']
                origin_w, origin_h = instance_num_to_shape[instance_number]
                x = x * self.img_size / origin_w
                y = y * self.img_size / origin_h

                # query to current
                instance_number = dicom_instance_numbers_to_idx[instance_number]
                idx = _level_to_idx[a['level']]
                has_labeled_levels.add(a['level'])
                if a['cond'] == 'Left Subarticular Stenosis':
                    keypoints_left[idx, 0] = int(x)
                    keypoints_left[idx, 1] = int(y)
                    keypoints_z_left[idx] = instance_number
                else:
                    keypoints_right[idx, 0] = int(x)
                    keypoints_right[idx, 1] = int(y)
                    keypoints_z_right[idx] = instance_number
            #has_labeled_levels = sorted(has_labeled_levels)
            keypoints = np.concatenate((keypoints_left, keypoints_right), axis=0)
            keypoints_z = np.concatenate((keypoints_z_left, keypoints_z_right), axis=0)
            mask = np.where(keypoints < 0, 0, 1)
            mask_z = np.where(keypoints_z < 0, 0, 1)

            if study_id not in gt_keypoint_info:
                gt_keypoint_info[study_id] = {}
            gt_keypoint_info[study_id][series_id] = {
                'keypoints': keypoints,
                'keypoints_z': keypoints_z,
                'mask': mask,
                'mask_z': mask_z,
                'has_labeled_levels': has_labeled_levels
            }
        return gt_keypoint_info

    def __getitem__(self, idx):
        item = self.samples[idx]

        study_id = int(item['study_id'])
        series_id = int(item['series_id'])

        keypoints_d = self.gt_keypoint_info[study_id][series_id]
        keypoints = keypoints_d['keypoints']
        keypoints_z = keypoints_d['keypoints_z']
        mask = keypoints_d['mask']
        mask_z = keypoints_d['mask_z']

        fns = glob(f'{self.volume_data_root}/{study_id}/{series_id}/*.png')
        fns = sorted(fns)
        img = self.load_volume(fns, None, bbox=None)

        if self.transform is not None:
            # to h,w,c
            img = img.transpose(1, 2, 0)
            augmented = self.transform(image=img, keypoints=keypoints)
            img = augmented['image']
            keypoints = augmented['keypoints']
            keypoints = np.array(keypoints)
            img = img.transpose(2, 0, 1)

        keypoints = 4 * keypoints / img.shape[1]
        keypoints_z = 16 * keypoints_z / img.shape[0]
        keypoints = np.concatenate((keypoints, keypoints_z), axis=1)

        # print(keypoints)
        # exit(0)

        mask = np.concatenate((mask, mask_z), axis=1)

        # print(keypoints.shape)
        # print(mask.shape)
        x = img.astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        # n, 512, 512
        # 96, 512, 512
        x = F.interpolate(x, size=(96, 256, 256)).squeeze()
        # print(x.shape)
        # exit(0)

        return {
            'img': x,
            'keypoints': keypoints.reshape(-1),
            'mask': mask.reshape((-1)),
            'study_id': study_id,
            'series_id': series_id,
        }


if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # print(df.iloc[0])
    # exit(0)
    # df = df[df['study_id'] == 11943292]
    aux_info = get_train_study_aux_info(data_root)

    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    create_dir(debug_dir)
    import albumentations as A

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        # A.OneOf([
        #     A.MotionBlur(blur_limit=5),
        #     A.MedianBlur(blur_limit=5),
        #     A.GaussianBlur(blur_limit=5),
        # ], p=AUG_PROB),

        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    transforms_val = A.Compose([
        # A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = RSNA24Dataset_KeyPoint_Axial_3D(data_root, aux_info, df, phase='valid',
                                           transform=transforms_train)

    print(len(dset))
    for d in dset:
        volumn = d['img']
        print('img: ', volumn.shape)
        keypoints = d['keypoints'].reshape(10, 3)
        mask = d['mask'].reshape(10, 3)
        if mask[:, 2].sum() < 5:
            print(d['study_id'])
            print(d['series_id'])
            print(keypoints)
            exit(0)

        # imgs = []
        # scale = volumn.shape[1] / 4
        # for p in keypoints:
        #     x, y, z = int(p[0] * scale), int(p[1] * scale), int(p[2] * volumn.shape[0] / 16)
        #     print('xyz: ', x, y, z)
        #     img = volumn[z]
        #     img = 255 * (img - img.min()) / (img.max() - img.min())
        #     img = np.array(img, dtype=np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
        #     cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #
        #     imgs.append(img)
        #
        # img_concat = np.concatenate(imgs, axis=0)
        # cv2.imwrite(f'{debug_dir}/keypoint_axial_3d.jpg', img_concat)
        # break
