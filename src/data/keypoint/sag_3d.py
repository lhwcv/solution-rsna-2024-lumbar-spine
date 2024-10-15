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

from src.config import G_IMG_SIZE
from src.utils.aux_info import get_train_study_aux_info
import torch.nn.functional as F
from train_scripts.v20.hard_list import hard_sag_t1_z_keypoints_study_id_list, hard_sag_t2_z_keypoints_study_id_list
sag_hard_z_list = hard_sag_t1_z_keypoints_study_id_list + hard_sag_t2_z_keypoints_study_id_list

_level_to_idx = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}


class RSNA24Dataset_KeyPoint_Sag_3D(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 img_size=512,
                 depth_3d=48,
                 img_size_3d=256,
                 ):
        without = [
            490052995,
            1261271580,
            2507107985,
            2626030939,
            2773343225,
            3109648055,
            3387993595,
            2492114990, 3008676218, 2780132468, 3637444890
        ]
        if phase != 'test':
            self.df = df[~df['study_id'].isin(without)]
        else:
            self.df = df
        self.aux_info = aux_info
        self.transform = transform

        self.phase = phase
        self.volume_data_root = f'{data_root}/train_images_preprocessed/'
        self.img_size = img_size
        self.depth_3d = depth_3d
        self.img_size_3d = img_size_3d

        self.gt_keypoint_info = {}
        for i, row in self.df.iterrows():
            study_id = row['study_id']
            t2_keypoints, t1_keypoints_left, t1_keypoints_right = \
                self.get_sag_keypoint_from_aux_info(study_id)
            self.gt_keypoint_info[study_id] = {
                't2_keypoints': t2_keypoints,
                't1_keypoints_left': t1_keypoints_left,
                't1_keypoints_right': t1_keypoints_right
            }

        print('[RSNA24Dataset_KeyPoint_Sag_3D] samples: ', len(self.df))

    def __len__(self):
        return len(self.df)

    def get_sag_keypoint_from_aux_info(self, study_id):
        info = self.aux_info[study_id]
        des_to_sid: dict = info['des_to_sid']
        sag_t1_sid = des_to_sid.get('Sagittal T1', [])
        sag_t2_sid = des_to_sid.get('Sagittal T2/STIR', [])
        sag_sids = sag_t1_sid.copy()
        sag_sids.extend(sag_t2_sid)
        assert len(sag_sids) >= 1
        if len(sag_sids) > 2:
            print(f'[warn] {study_id} has {len(sag_sids)} sag series')

        label_info = info['aux_infos']
        dataframes = {
            'series_id': [],
            'instance_number': [],
            'x': [],
            'y': [],
            'level': [],
            'cond': [],
        }
        for cond, coord_labels in label_info.items():
            for a in coord_labels:
                if a['series_id'] not in sag_sids:
                    continue
                dataframes['series_id'].append(a['series_id'])
                dataframes['instance_number'].append(a['instance_number'])
                dataframes['x'].append(a['x'])
                dataframes['y'].append(a['y'])
                dataframes['level'].append(a['level'])
                dataframes['cond'].append(cond)
        dataframes = pd.DataFrame(dataframes)
        ## sag t1
        t1_keypoints_left = -np.ones((5, 3), dtype=np.float32)
        t1_keypoints_right = -np.ones((5, 3), dtype=np.float32)

        if len(sag_t1_sid) > 0:
            series_id = sag_t1_sid[0]
            sub_dfs = dataframes[dataframes['series_id'] == series_id]
            labeled_keypoints = sub_dfs.to_dict('records')
            meta_fn = f'{self.volume_data_root}/{study_id}/{series_id}/meta_info_2.pkl'
            meta = pickle.load(open(meta_fn, 'rb'))
            instance_num_to_shape = meta['instance_num_to_shape']
            dicom_instance_numbers_to_idx = {}
            for i, ins in enumerate(meta['dicom_instance_numbers']):
                dicom_instance_numbers_to_idx[ins] = i

            for a in labeled_keypoints:
                instance_number = a['instance_number']
                x, y = a['x'], a['y']
                origin_w, origin_h = instance_num_to_shape[instance_number]
                x = x * self.img_size / origin_w
                y = y * self.img_size / origin_h

                # query to current
                z = dicom_instance_numbers_to_idx[instance_number]
                idx = _level_to_idx[a['level']]
                if a['cond'] == 'Left Neural Foraminal Narrowing':
                    t1_keypoints_left[idx, 0] = int(x)
                    t1_keypoints_left[idx, 1] = int(y)
                    t1_keypoints_left[idx, 2] = z
                else:
                    t1_keypoints_right[idx, 0] = int(x)
                    t1_keypoints_right[idx, 1] = int(y)
                    t1_keypoints_right[idx, 2] = z

        # print(t1_keypoints_right)
        ## sag t2
        t2_keypoints = -np.ones((5, 3), dtype=np.float32)
        if len(sag_t2_sid) > 0:
            series_id = sag_t2_sid[0]
            sub_dfs = dataframes[dataframes['series_id'] == series_id]
            labeled_keypoints = sub_dfs.to_dict('records')
            meta_fn = f'{self.volume_data_root}/{study_id}/{series_id}/meta_info_2.pkl'
            meta = pickle.load(open(meta_fn, 'rb'))
            instance_num_to_shape = meta['instance_num_to_shape']
            dicom_instance_numbers_to_idx = {}
            for i, ins in enumerate(meta['dicom_instance_numbers']):
                dicom_instance_numbers_to_idx[ins] = i

            for a in labeled_keypoints:
                instance_number = a['instance_number']
                x, y = a['x'], a['y']
                origin_w, origin_h = instance_num_to_shape[instance_number]
                x = x * self.img_size / origin_w
                y = y * self.img_size / origin_h

                # query to current
                z = dicom_instance_numbers_to_idx[instance_number]
                idx = _level_to_idx[a['level']]
                t2_keypoints[idx, 0] = int(x)
                t2_keypoints[idx, 1] = int(y)
                t2_keypoints[idx, 2] = z
        return t2_keypoints, t1_keypoints_left, t1_keypoints_right

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

    def get_fns(self, study_id, sids):
        if self.phase != 'train':
            # TODO use multi when test
            id = sids[0]
        else:
            id = random.choice(sids)
        fns = glob(f'{self.volume_data_root}/{study_id}/{id}/*.png')
        fns = sorted(fns)
        return fns, id

    def gen_slice_index_Sagittal(self, length, sag_total_slices):
        step = length / sag_total_slices
        s = (sag_total_slices - 2) / 2.0
        st = length / 2.0 - s * step
        end = length + 0.0001
        slice_indexes = []
        for i in np.arange(st, end, step):
            inx = max(0, int((i - 0.5001).round()))
            slice_indexes.append(inx)
        return slice_indexes

    def __getitem__(self, idx):
        d = self.get_keypoint_item(idx)
        # {
        #     's_t1': s_t1,
        #     's_t2': s_t2,
        #     'keypoints': keypoints,
        #     'mask': mask,
        #     'label': label
        # }
        ret = {
            'keypoints': d['keypoints'].reshape(-1),
            'mask': d['mask'].reshape((-1)),
            'label': d['label'],
            'study_ids': d['study_ids']
        }
        s_t1 = d['s_t1']
        s_t1 = s_t1.astype(np.float32)
        s_t1 = torch.from_numpy(s_t1).unsqueeze(0).unsqueeze(0)
        s_t1 = F.interpolate(s_t1, size=(self.depth_3d, self.img_size_3d, self.img_size_3d)).squeeze(0)

        s_t2 = d['s_t2']
        s_t2 = s_t2.astype(np.float32)
        s_t2 = torch.from_numpy(s_t2).unsqueeze(0).unsqueeze(0)
        s_t2 = F.interpolate(s_t2, size=(self.depth_3d, self.img_size_3d, self.img_size_3d)).squeeze(0)

        x = torch.cat((s_t1, s_t2), dim=0)
        ret['img'] = x
        return ret

    def get_keypoint_item(self, idx, sag_total_slices=-1):
        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        label = item[1:].values.astype(np.int64)
        # label = label[:15]  # without Axial (Stenosis)
        keypoints_d = self.gt_keypoint_info[study_id]

        t2_keypoints = keypoints_d['t2_keypoints']
        t1_keypoints_left = keypoints_d['t1_keypoints_left']
        t1_keypoints_right = keypoints_d['t1_keypoints_right']

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

        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        has_t1 = True
        if 'Sagittal T1' not in des_to_sid.keys():
            has_t1 = False
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10 if sag_total_slices == -1 else sag_total_slices,
                             G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            label[5:15] = -100
        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T1'])
            if sag_total_slices != -1:
                index = self.gen_slice_index_Sagittal(len(fns), sag_total_slices)
            else:
                index = None

            s_t1 = self.load_volume(fns, index)

        # Sagittal T2/STIR
        has_t2 = True
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            has_t2 = False
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10 if sag_total_slices == -1 else sag_total_slices,
                             G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            label[:5] = -100

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T2/STIR'])
            if sag_total_slices != -1:
                index = self.gen_slice_index_Sagittal(len(fns), sag_total_slices)
            else:
                index = None
            s_t2 = self.load_volume(fns, index)

        c1 = s_t1.shape[0]
        c2 = s_t2.shape[0]

        if self.transform is not None:
            # same aug
            img = np.concatenate((s_t1, s_t2), axis=0)
            # to h,w,c
            img = img.transpose(1, 2, 0)
            augmented = self.transform(image=img, keypoints=keypoints_xy)
            img = augmented['image']
            keypoints_xy = augmented['keypoints']
            keypoints_xy = np.array(keypoints_xy)
            img = img.transpose(2, 0, 1)
            s_t1 = img[:c1]
            s_t2 = img[c1:]

        keypoints_xy = 4 * keypoints_xy / self.img_size
        keypoints_z[:5] = 16 * keypoints_z[:5] / c2
        keypoints_z[5:] = 16 * keypoints_z[5:] / c1

        keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)
        mask = np.concatenate((mask_xy, mask_z), axis=1)

        return {
            's_t1': s_t1,
            's_t2': s_t2,
            'keypoints': keypoints,
            'mask': mask,
            'label': label,
            'study_ids': study_id,
        }


class RSNA24Dataset_Sag_Cls_Use_GT_Point(RSNA24Dataset_KeyPoint_Sag_3D):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 z_imgs=3,
                 saggital_fixed_slices=-1,
                 xy_use_model_pred=False,
                 z_use_specific=False,
                 img_size=512):
        super(RSNA24Dataset_Sag_Cls_Use_GT_Point, self).__init__(
            data_root,
            aux_info,
            df,
            phase=phase,
            transform=transform,
            img_size=img_size
        )
        self.z_imgs = z_imgs
        self.saggital_fixed_slices = saggital_fixed_slices
        self.z_use_specific = z_use_specific

        self.xy_use_model_pred = xy_use_model_pred
        if xy_use_model_pred:
            keypoints_infos = pickle.load(
                open(f'{data_root}/study_id_to_pred_keypoints.pkl', 'rb'))
            self.keypoints_infos = {}
            for k, v in keypoints_infos.items():
                self.keypoints_infos[int(k)] = v

        self.pred_sag_keypoints_infos_3d = pickle.load(
            open(f'{data_root}/v2_sag_3d_keypoints_en3.pkl', 'rb'))

        easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()
        self.easy_study_ids = set(easy_study_ids)

    def crop_out_by_keypoints(self, volume, keypoints_xyz,
                              z_imgs=3, crop_size=128):
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

            # v = v.transpose(1, 2, 0)
            # v = cv2.resize(v, (128, 128), interpolation=cv2.INTER_CUBIC)
            # v = v.transpose(2, 0, 1)
            sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        return volume_crop, None

    def __getitem__(self, idx):

        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        label = item[1:].values.astype(np.int64)
        # label = label[:15]  # without Axial (Stenosis)
        keypoints_d = self.gt_keypoint_info[study_id]

        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        sag_total_slices = -1
        has_t1 = True
        if 'Sagittal T1' not in des_to_sid.keys():
            has_t1 = False
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10 if sag_total_slices == -1 else sag_total_slices,
                             G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            label[5:15] = -100
        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T1'])
            if sag_total_slices != -1:
                index = self.gen_slice_index_Sagittal(len(fns), sag_total_slices)
            else:
                index = None

            s_t1 = self.load_volume(fns, index)

        # Sagittal T2/STIR
        has_t2 = True
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            has_t2 = False
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10 if sag_total_slices == -1 else sag_total_slices,
                             G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
            label[:5] = -100

        else:
            fns, sid = self.get_fns(study_id, des_to_sid['Sagittal T2/STIR'])
            if sag_total_slices != -1:
                index = self.gen_slice_index_Sagittal(len(fns), sag_total_slices)
            else:
                index = None
            s_t2 = self.load_volume(fns, index)

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

        if self.phase == 'train' and random.random() < 0.5:
            # gt keypoints
            t2_keypoints = keypoints_d['t2_keypoints']
            t1_keypoints_left = keypoints_d['t1_keypoints_left']
            t1_keypoints_right = keypoints_d['t1_keypoints_right']

        if study_id not in self.easy_study_ids:
            t2_keypoints = keypoints_d['t2_keypoints']
            t1_keypoints_left = keypoints_d['t1_keypoints_left']
            t1_keypoints_right = keypoints_d['t1_keypoints_right']

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

        if self.phase == 'train' and random.random() < 0.5:
            noise1 = np.random.uniform(-5.0, 5.0, keypoints_xy.shape)
            keypoints_xy = keypoints_xy + noise1

        if self.transform is not None:
            # same aug
            img = np.concatenate((s_t1, s_t2), axis=0)
            # to h,w,c
            img = img.transpose(1, 2, 0)
            augmented = self.transform(image=img, keypoints=keypoints_xy)
            img = augmented['image']
            keypoints_xy = augmented['keypoints']
            keypoints_xy = np.array(keypoints_xy)
            img = img.transpose(2, 0, 1)
            s_t1 = img[:c1]
            s_t2 = img[c1:]

        keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)
        mask = np.concatenate((mask_xy, mask_z), axis=1)

        s_t2_keypoints = keypoints[:5]
        s_t1_left_keypoints = keypoints[5:10]
        s_t1_right_keypoints = keypoints[10:15]

        # if self.xy_use_model_pred:
        #     item = self.df.iloc[idx]
        #     study_id = int(item['study_id'])
        #     pred_keypoints = self.keypoints_infos[study_id]
        #     pred_s_t1_keypoints = pred_keypoints[0]
        #     pred_s_t2_keypoints = pred_keypoints[1]
        #     s_t2_keypoints[:, :2] = pred_s_t2_keypoints
        #     s_t1_left_keypoints[:, :2] = pred_s_t1_keypoints
        #     s_t1_right_keypoints[:, :2] = pred_s_t1_keypoints

        if self.z_use_specific:
            assert self.saggital_fixed_slices != -1
            assert self.saggital_fixed_slices >= 2 * self.z_imgs
            # center
            s_t2_keypoints[:, 2] = c2 // 2
            #
            s_t1_left_keypoints[:, 2] = c1 // 2 + self.saggital_fixed_slices // 2
            s_t1_right_keypoints[:, 2] = c1 // 2 - self.saggital_fixed_slices // 2

        crop_size = 128
        s_t2, _ = self.crop_out_by_keypoints(s_t2, s_t2_keypoints,
                                             z_imgs=self.z_imgs, crop_size=crop_size)

        s_t1_left, _ = self.crop_out_by_keypoints(s_t1, s_t1_left_keypoints,
                                                  z_imgs=self.z_imgs, crop_size=crop_size)
        #
        s_t1_right, _ = self.crop_out_by_keypoints(s_t1, s_t1_right_keypoints,
                                                   z_imgs=self.z_imgs, crop_size=crop_size)

        # x = np.concatenate([s_t2, s_t1_left, s_t1_right], axis=0)
        # x = x.astype(np.float32)
        # cond = [0] * 5 + [1] * 5 + [2] * 5
        # cond = np.array(cond, dtype=np.int64)

        label = label[5:15]
        x = np.concatenate([s_t1_left, s_t1_right], axis=0)
        x = x.astype(np.float32)
        cond = [1] * 5 + [2] * 5
        cond = np.array(cond, dtype=np.int64)

        # label = label[:5]
        # x = np.concatenate([s_t2], axis=0)
        # x = x.astype(np.float32)
        # cond = [0] * 5
        # cond = np.array(cond, dtype=np.int64)

        return {
            'img': x,
            # 's_t2': s_t2,
            # 's_t1_left': s_t1_left,
            # 's_t1_right': s_t1_right,
            'study_ids': study_id,
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

    # df = df[df['study_id'] == 59576878]
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

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    transforms_val = A.Compose([
        # A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = RSNA24Dataset_Sag_Cls_Use_GT_Point(data_root, aux_info, df, phase='valid',
                                              transform=transforms_train,
                                              z_imgs=7,
                                              saggital_fixed_slices=16,
                                              z_use_specific=True,
                                              xy_use_model_pred=True)
    print(len(dset))
    for d in dset:
        print('x: ', d['img'].shape)
        s_t2 = d['s_t2']
        s_t1_left = d['s_t1_left']
        s_t1_right = d['s_t1_right']
        print('img: ', d['img'].shape)
        print('s_t2: ', s_t2.shape)
        print('s_t1_left: ', s_t1_left.shape)
        print('s_t1_right: ', s_t1_right.shape)

        cv2.imwrite(f'{debug_dir}/v2_s_t2.jpg', convert_to_cv2_img(s_t2))
        cv2.imwrite(f'{debug_dir}/v2_s_t1_left.jpg', convert_to_cv2_img(s_t1_left))
        cv2.imwrite(f'{debug_dir}/v2_s_t1_right.jpg', convert_to_cv2_img(s_t1_right))
        break
    # dset = RSNA24Dataset_KeyPoint_Sag_3D(data_root, aux_info, df, phase='valid',
    #                                      transform=transforms_train)
    #
    # print(len(dset))
    # for d in dset:
    #     volumn = d['s_t1']
    #     print('img: ', volumn.shape)
    #     keypoints = d['keypoints'].reshape(15, 3)
    #     # t1 keypoints
    #     keypoints = keypoints[5:]
    #     imgs = []
    #     scale = volumn.shape[1] / 4
    #     for p in keypoints:
    #         x, y, z = int(p[0] * scale), int(p[1] * scale), int(p[2] * volumn.shape[0] / 16)
    #         print('t1 xyz: ', x, y, z)
    #         img = volumn[z]
    #
    #         img = 255 * (img - img.min()) / (img.max() - img.min())
    #         img = np.array(img, dtype=np.uint8)
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #         img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
    #         cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #         imgs.append(img)
    #
    #     img_concat = np.concatenate(imgs, axis=0)
    #     cv2.imwrite(f'{debug_dir}/keypoint_sag_3d_t1.jpg', img_concat)
    #
    #     volumn = d['s_t2']
    #     print('img: ', volumn.shape)
    #     keypoints = d['keypoints'].reshape(15, 3)
    #     # t1 keypoints
    #     keypoints = keypoints[:5]
    #     imgs = []
    #     scale = volumn.shape[1] / 4
    #     for p in keypoints:
    #         x, y, z = int(p[0] * scale), int(p[1] * scale), int(p[2] * volumn.shape[0] / 16)
    #         print('t2 xyz: ', x, y, z)
    #         img = volumn[z]
    #
    #         img = 255 * (img - img.min()) / (img.max() - img.min())
    #         img = np.array(img, dtype=np.uint8)
    #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #         img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
    #         cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #         imgs.append(img)
    #
    #     img_concat = np.concatenate(imgs, axis=0)
    #     cv2.imwrite(f'{debug_dir}/keypoint_sag_3d_t2.jpg', img_concat)
    #     print('write done')
    #     break
