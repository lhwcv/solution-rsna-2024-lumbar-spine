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

_level_to_idx = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}


class RSNA24Dataset_KeyPoint_Sag_2d(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 target_conditions=['Spinal Canal Stenosis']
                 ):

        self.df = df
        self.aux_info = aux_info
        self.transform = transform

        self.phase = phase
        self.img_size = 512
        # self.volume_data_root = f'{data_root}/train_images_preprocessed_{self.img_size}/'
        self.volume_data_root = f'{data_root}/train_images_preprocessed'
        self.samples = []
        for i, row in df.iterrows():
            study_id = row['study_id']
            label_info = self.aux_info[study_id]['aux_infos']
            dataframes = {
                'series_id': [],
                'instance_number': [],
                'x': [],
                'y': [],
                'level': []
            }
            for cond, coord_labels in label_info.items():
                if cond not in target_conditions:
                    continue
                # if cond in ['Left Subarticular Stenosis']:
                #     continue
                # if cond in ['Right Subarticular Stenosis']:
                #     continue
                for a in coord_labels:
                    dataframes['series_id'].append(a['series_id'])
                    dataframes['instance_number'].append(a['instance_number'])
                    dataframes['x'].append(a['x'])
                    dataframes['y'].append(a['y'])
                    dataframes['level'].append(a['level'])

            dataframes = pd.DataFrame(dataframes)
            for series_id in dataframes['series_id'].unique():
                sub_dfs = dataframes[dataframes['series_id'] == series_id]
                for _, g in sub_dfs.groupby('instance_number'):
                    # convert g to list of dict
                    g = g.to_dict('records')
                    # print(g)
                    # exit(0)
                    self.samples.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'labeled_keypoints': g
                    })

        print('[RSNA24Dataset_KeyPoint_Sag_2d] samples: ', len(self.samples))
        print(self.samples[0])

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

    def __getitem__(self, idx):

        item = self.samples[idx]
        study_id = int(item['study_id'])
        series_id = int(item['series_id'])

        # print('study_id: ', study_id)

        meta_fn = f'{self.volume_data_root}/{study_id}/{series_id}/meta_info_2.pkl'
        meta = pickle.load(open(meta_fn, 'rb'))
        instance_num_to_shape = meta['instance_num_to_shape']
        dicom_instance_numbers_to_idx = {}
        for i, ins in enumerate(meta['dicom_instance_numbers']):
            dicom_instance_numbers_to_idx[ins] = i

        keypoints = np.zeros((5, 2), dtype=np.float32)

        instance_numbers = set()
        for a in item['labeled_keypoints']:
            instance_number = a['instance_number']
            instance_numbers.add(instance_number)
            x, y = a['x'], a['y']
            origin_w, origin_h = instance_num_to_shape[instance_number]
            x = x * self.img_size / origin_w
            y = y * self.img_size / origin_h

            idx = _level_to_idx[a['level']]
            keypoints[idx, 0] = int(x)
            keypoints[idx, 1] = int(y)

        # print(keypoints)
        assert len(instance_numbers) == 1

        instance_number = list(instance_numbers)[0]
        instance_number = dicom_instance_numbers_to_idx[instance_number]
        # instance_numbers = [instance_number - 1, instance_number, instance_number + 1]
        instance_numbers = [instance_number]

        fns = glob(f'{self.volume_data_root}/{study_id}/{series_id}/*.png')
        fns = sorted(fns)
        for i in range(len(instance_numbers)):
            if instance_numbers[i] < 0:
                instance_numbers[i] = 0
            if instance_numbers[i] > len(fns) - 1:
                instance_numbers[i] = len(fns) - 1

        img = self.load_volume(fns, instance_numbers, bbox=None)
        mask = np.where(keypoints == 0, 0, 1)

        if self.transform is not None:
            # to h,w,c
            img = img.transpose(1, 2, 0)
            augmented = self.transform(image=img, keypoints=keypoints)
            img = augmented['image']
            keypoints = augmented['keypoints']
            keypoints = np.array(keypoints)
            img = img.transpose(2, 0, 1)

        keypoints = keypoints / 128.0
        x = img.astype(np.float32)

        return {
            'img': x,
            'keypoints': keypoints.reshape(-1),
            'mask': mask.reshape((-1))
        }


if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # df = df[df['study_id']==1667735473]
    df = df.sample(frac=1)
    aux_info = get_train_study_aux_info(data_root)

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
        A.HorizontalFlip(p=AUG_PROB),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = RSNA24Dataset_KeyPoint_Sag_2d(data_root, aux_info, df, phase='valid',
                                         transform=transforms_train)

    print(len(dset))
    for d in dset:
        img = d['img']
        print('mask: ', d['mask'])
        img = np.concatenate([img, img, img], axis=0)
        img = img.transpose(1, 2, 0)
        img = np.array(img, dtype=np.uint8)
        print('img: ', img.shape)
        keypoints = d['keypoints'].reshape(5, 2)

        for p in keypoints:
            x, y = int(p[0] * 128), int(p[1] * 128)
            print(x, y)

            img = cv2.circle(img.copy(), (x, y), 5, (0, 0, 255), 9)

        cv2.imwrite(f'{debug_dir}/keypoint_sag_2d.jpg', img)
        break
