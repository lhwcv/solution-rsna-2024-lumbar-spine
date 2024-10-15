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
import torch.nn.functional as F
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
            group_idxs = gen_level_group_idxs(pred_z, depth, ext)
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

class RSNA24Dataset_Cls_V4(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 transforms_axial=None,
                 in_channels=5,
                 axial_crop_xy_size = -1,
                 axial_margin_extend = 1.0,
                 ):
        train_without = [2492114990, 3008676218, 2780132468, 3637444890]
        if phase == 'train':
            self.df = df[~df['study_id'].isin(train_without)]
        else:
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

        self.dict_axial_group_idxs = build_axial_group_idxs(data_root,
                                                            img_size=G_IMG_SIZE,
                                                            ext=axial_margin_extend)

        self.img_size = 512

        self.sag_total_slices = 10
        self.sag_t2_index_range = [3, 8]
        self.sag_t1_left_index_range = [5, 10]
        self.sag_t1_right_index_range = [0, 5]
        self.random_shift_sag_keypoint_prob = 0.0
        self.axial_imgs_one_level = 6

        self.axial_with_xy_crop = True if axial_crop_xy_size!=-1 else False
        self.axial_crop_xy_size = axial_crop_xy_size


    def __len__(self):
        return len(self.df)

    def load_volume(self, fns, index=None, bbox=None):
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
            sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        return volume_crop, None  # att_mask



    def __getitem__(self, idx):

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
        # if self.phase=='train' and random.random() < self.random_shift_sag_keypoint_prob:
        #     noise1 = np.random.uniform(-3.0, 3.0, s_t1_keypoints.shape)
        #     s_t1_keypoints += noise1
        #     noise2 = np.random.uniform(-3.0, 3.0, s_t1_keypoints.shape)
        #     s_t2_keypoints += noise2


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
            s_t1 = self.load_volume(fns)

        # Sagittal T2/STIR
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((self.sag_total_slices, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            label[:5] = 0
            s_t2_keypoints = np.zeros((5, 2), dtype=np.int64)
            s_t2_keypoints[:] = G_IMG_SIZE // 2

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T2/STIR'])
            s_t2 = self.load_volume(fns, bbox=None)

        if self.phase == 'train':
            s_t1 = self.drop_img(s_t1, prob=0.1)
            s_t2 = self.drop_img(s_t2, prob=0.1)

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

            #s_t1 = s_t1.transpose(2, 0, 1)
            #s_t2 = s_t2.transpose(2, 0, 1)
        else:
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)

        s_t2_keypoints = np.array(s_t2_keypoints)
        keypoints = np.zeros((5, 3))
        keypoints[:, :2] = s_t2_keypoints / 512.0

        s_t1 = s_t1.astype(np.float32)
        s_t1 = torch.from_numpy(s_t1).unsqueeze(0).unsqueeze(0)
        s_t2 = s_t2.astype(np.float32)
        s_t2 = torch.from_numpy(s_t2).unsqueeze(0).unsqueeze(0)

        s_t1 = F.interpolate(s_t1, size=(512, 512, 32)).squeeze(0)
        s_t2 = F.interpolate(s_t2, size=(512, 512, 32)).squeeze(0)
        # 2, h, w, d
        s_img = torch.cat((s_t1, s_t2), dim=0)

        return {
            'img': s_img,
            'label': label,
            'keypoints': keypoints
        }


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

    dset = RSNA24Dataset_Cls_V4(data_root, aux_info, df, phase='valid',
                                               transform=transforms_train,
                                               transforms_axial=transforms_axial)
    print(len(dset))
    for d in dset:
        print('x: ', d['img'].shape)
        print(d['keypoints'])
        break