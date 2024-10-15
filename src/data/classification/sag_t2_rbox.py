# -*- coding: utf-8 -*-
import json
import math
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
from src.data.keypoint.sag_3d import RSNA24Dataset_KeyPoint_Sag_3D
from src.data.keypoint.axial_3d import RSNA24Dataset_KeyPoint_Axial_3D
from src.data.classification.v2_all_view import build_axial_group_idxs

_level_to_idx = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}

def sort_points(pts):
    sorted_pts = sorted(pts, key=lambda pt: pt[0])
    left_pts = sorted(sorted_pts[:2], key=lambda pt: pt[1])
    right_pts = sorted(sorted_pts[2:], key=lambda pt: pt[1])

    return [ left_pts[0], right_pts[0], right_pts[1], left_pts[1] ]

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
        }

    return saggital_t2_rbox_info


class RSNA24Dataset_Sag_Axial_Cls(Dataset):
    def __init__(self,
                 data_root,
                 aux_info,
                 df,
                 phase='train',
                 transform=None,
                 z_imgs=3,
                 axial_z_imgs=5,
                 saggital_fixed_slices=-1,
                 xy_use_model_pred=False,
                 z_use_specific=False,
                 img_size=512,
                 with_sag=True,
                 with_axial=False,
                 axial_transform=None):
        super(RSNA24Dataset_Sag_Axial_Cls, self).__init__()

        self.sag_keypoint_dset = RSNA24Dataset_KeyPoint_Sag_3D(
            data_root,
            aux_info,
            df,
            img_size=img_size)

        print('build_saggital_t2_rbox ..')
        self.saggital_t2_rbox_info = build_saggital_t2_rbox(data_root, img_size)
        print('build_saggital_t2_rbox done')

        self.axial_keypoint_dset = RSNA24Dataset_KeyPoint_Axial_3D(
            data_root,
            aux_info,
            df,
            img_size=img_size)

        self.dict_axial_group_idxs = build_axial_group_idxs(data_root,
                                                            img_size=G_IMG_SIZE,
                                                            ext=1.0)

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
        self.df = df[~df['study_id'].isin(without)]
        self.aux_info = aux_info
        self.transform = transform
        self.phase = phase
        self.volume_data_root = f'{data_root}/train_images_preprocessed/'
        self.img_size = img_size

        self.z_imgs = z_imgs
        self.saggital_fixed_slices = saggital_fixed_slices
        self.z_use_specific = z_use_specific
        self.xy_use_model_pred = xy_use_model_pred

        self.with_sag = with_sag
        self.with_axial = with_axial
        self.axial_transform = axial_transform
        self.axial_z_imgs = axial_z_imgs

        if xy_use_model_pred:
            keypoints_infos = pickle.load(
                open(f'{data_root}/study_id_to_pred_keypoints.pkl', 'rb'))
            self.keypoints_infos = {}
            for k, v in keypoints_infos.items():
                self.keypoints_infos[int(k)] = v

        self.pred_sag_keypoints_infos_3d = pickle.load(
            open(f'{data_root}/v2_sag_3d_keypoints_oof.pkl', 'rb'))

        easy_study_ids = pd.read_csv(f'{data_root}/easy_list_0813.csv')['study_id'].tolist()
        self.easy_study_ids = set(easy_study_ids)

    def __len__(self):
        return len(self.df)

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

            # v = v.transpose(1, 2, 0)
            # v = cv2.resize(v, (128, 128), interpolation=cv2.INTER_CUBIC)
            # v = v.transpose(2, 0, 1)

            if transform is not None:
                v = v.transpose(1, 2, 0)
                v = transform(image=v)['image']
                v = v.transpose(2, 0, 1)

            sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        return volume_crop, None

    def crop_out_by_rbox(self, volume,
                         rboxes,
                         keypoints_z,
                         z_imgs=3,
                         crop_size=128,
                         transform=None):
        sub_volume_list = []
        ii = 0
        for box, z in zip(rboxes, keypoints_z):
            z = int(z)
            # no z
            if z < 0:
                v = np.zeros((z_imgs, crop_size, crop_size), dtype=volume.dtype)
                sub_volume_list.append(v)
                continue

            z0 = z - z_imgs // 2
            z1 = z + z_imgs // 2 + 1
            if z0 < 0:
                z0 = 0
                z1 = z_imgs
            if z1 > volume.shape[0]:
                z0 = volume.shape[0] - z_imgs
                z1 = volume.shape[0]
            v = volume[z0: z1].copy()

            v = v.transpose(1, 2, 0)

            mR = cv2.minAreaRect(box.copy())
            box_width, box_height = mR[1]
            if box_width < box_height:
                box_width, box_height = box_height, box_width
            w, h = box_width, box_height
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32')
            M = cv2.getPerspectiveTransform(box, dst)
            cropped = cv2.warpPerspective(v, M, (int(w), int(h)))
            cropped = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC)

            v = transform(image=cropped)['image']
            v = v.transpose(2, 0, 1)

            sub_volume_list.append(v)
            # if ii == 4:
            #     print(box)
            # ii +=1


        volume_crop = np.array(sub_volume_list)
        return volume_crop, None

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

    def get_fns_axial(self, study_id, sids):
        if self.phase != 'train':
            # TODO use multi when test
            # id = sids[0]
            # max_id = -1
            max_fns = []
            # max_n = 0
            multiple_input = 0
            for id in sids:
                fns = glob(f'{self.volume_data_root}/{study_id}/{id}/*.png')
                max_fns.append(sorted(fns))
            if len(sids) > 1:
                multiple_input = 1

                return max_fns, list(sids), multiple_input
            else:
                return max_fns[0], sids[0], multiple_input
        else:
            id = random.choice(sids)
            fns = glob(f'{self.volume_data_root}/{study_id}/{id}/*.png')
            fns = sorted(fns)
            return fns, id, 0

    def __getitem__(self, idx):

        item = self.df.iloc[idx]
        study_id = int(item['study_id'])
        label = item[1:].values.astype(np.int64)
        # label = label[:15]  # without Axial (Stenosis)
        #

        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        ret_dict = {
            'study_ids': study_id,
            'label': label,
        }

        if self.with_sag:
            # Sagittal T1
            sag_total_slices = -1

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

            c2 = s_t2.shape[0]

            t2_sid = des_to_sid['Sagittal T2/STIR'][0]
            rboxes  = self.saggital_t2_rbox_info[study_id][t2_sid]['rboxes']
            keypoints = self.saggital_t2_rbox_info[study_id][t2_sid]['keypoints']
            # t2_img = s_t2[c2//2]
            # t2_img = cv2.cvtColor(t2_img, cv2.COLOR_GRAY2BGR)
            # for box, pts in zip(rboxes, keypoints):
            #     box = np.asarray(box, dtype=int)
            #     cv2.drawContours(t2_img, [box], 0, (0, 0, 255), 2)
            #     p0, p1 = pts[0], pts[1]
            #     cv2.circle(t2_img, (int(p0[0]), int(p0[1])), 5, (0, 0, 255), 3)
            #     cv2.circle(t2_img, (int(p1[0]), int(p1[1])), 5, (0, 0, 255), 3)
            #     cv2.line(t2_img,
            #              (int(p0[0]), int(p0[1])),
            #              (int(p1[0]), int(p1[1])),
            #              (0, 255, 255), 3)
            #
            # cv2.imwrite(f'{data_root}/debug_dir/001_t2_img.jpg', t2_img)
            # exit(0)

            keypoints_d = self.sag_keypoint_dset.gt_keypoint_info[study_id]
            t2_keypoints = keypoints_d['t2_keypoints']
            keypoints_z = np.concatenate((t2_keypoints[:, 2:]), axis=0)


            crop_size = 128
            s_t2, _ = self.crop_out_by_rbox(s_t2, rboxes, keypoints_z,
                                            z_imgs=self.z_imgs, crop_size=crop_size,
                                            transform=self.transform)


            x = np.concatenate([s_t2], axis=0)
            x = x.astype(np.float32)

            ret_dict['img'] = x

        if self.with_axial:

            # Axial
            fns_all, sid_all, multiple_input = self.get_fns_axial(study_id, des_to_sid['Axial T2'])

            if multiple_input == 0:
                fns, sid = self.get_fns(study_id, des_to_sid['Axial T2'])
                axial_t2 = self.load_volume(fns)

                pred_xy = self.dict_axial_group_idxs[study_id][sid]['pred_xy'].reshape(2, 5, 2)
                pred_z = self.dict_axial_group_idxs[study_id][sid]['pred_z'].reshape(2, 5, 1)

                gt_keypoints = self.axial_keypoint_dset.gt_keypoint_info[study_id][sid]
                gt_xy = gt_keypoints['keypoints'].reshape(2, 5, 2)
                gt_z = gt_keypoints['keypoints_z'].reshape(2, 5, 1)

                if self.phase == 'train' and random.random() < 0.5:
                    keypoints = np.concatenate([gt_xy, gt_z], axis=2)
                else:
                    keypoints = np.concatenate([pred_xy, pred_z], axis=2)

                left_keypoints = keypoints[0]
                right_keypoints = keypoints[1]

                # print(keypoints)
                axial_t2_left, _ = self.crop_out_by_keypoints(axial_t2, left_keypoints,
                                                              z_imgs=self.axial_z_imgs,
                                                              crop_size=160,
                                                              transform=self.axial_transform
                                                              )
                axial_t2_right, _ = self.crop_out_by_keypoints(axial_t2, right_keypoints,
                                                               z_imgs=self.axial_z_imgs,
                                                               crop_size=160,
                                                               transform=self.axial_transform
                                                               )
                axial_imgs = np.concatenate([axial_t2_left, axial_t2_right], axis=0)
                axial_imgs = axial_imgs.astype(np.float32)
                ret_dict['axial_imgs'] = axial_imgs
            else:
                tmp_imgs = []
                for fns, sid in zip(fns_all, sid_all):
                    axial_t2 = self.load_volume(fns)
                    pred_xy = self.dict_axial_group_idxs[study_id][sid]['pred_xy'].reshape(2, 5, 2)
                    pred_z = self.dict_axial_group_idxs[study_id][sid]['pred_z'].reshape(2, 5, 1)

                    gt_keypoints = self.axial_keypoint_dset.gt_keypoint_info[study_id][sid]
                    gt_xy = gt_keypoints['keypoints'].reshape(2, 5, 2)
                    gt_z = gt_keypoints['keypoints_z'].reshape(2, 5, 1)

                    if self.phase == 'train' and random.random() < 0.5:
                        keypoints = np.concatenate([gt_xy, gt_z], axis=2)
                    else:
                        keypoints = np.concatenate([pred_xy, pred_z], axis=2)

                    left_keypoints = keypoints[0]
                    right_keypoints = keypoints[1]

                    # print(keypoints)
                    axial_t2_left, _ = self.crop_out_by_keypoints(axial_t2, left_keypoints,
                                                                  z_imgs=self.axial_z_imgs,
                                                                  crop_size=160,
                                                                  transform=self.axial_transform
                                                                  )
                    axial_t2_right, _ = self.crop_out_by_keypoints(axial_t2, right_keypoints,
                                                                   z_imgs=self.axial_z_imgs,
                                                                   crop_size=160,
                                                                   transform=self.axial_transform
                                                                   )
                    # 10, 3, 128, 128
                    axial_imgs = np.concatenate([axial_t2_left, axial_t2_right], axis=0)
                    tmp_imgs.append(axial_imgs)
                tmp_imgs = np.concatenate(tmp_imgs, axis=0)  # 10*num_series,3,h,w
                tmp_imgs = tmp_imgs.astype(np.float32)
                ret_dict['axial_imgs'] = tmp_imgs

        if self.with_sag and self.with_axial:
            cond = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
            cond = np.array(cond, dtype=np.int64)
            ret_dict['cond'] = cond

        if self.with_sag and not self.with_axial:
            # label = label[:15]
            # cond = [0] * 5 + [1] * 5 + [2] * 5
            label = label[:5]
            cond = [0] * 5

            cond = np.array(cond, dtype=np.int64)
            ret_dict['cond'] = cond

        if not self.with_sag and self.with_axial:
            label = label[15:]
            cond = [3] * 5 + [4] * 5
            cond = np.array(cond, dtype=np.int64)
            ret_dict['cond'] = cond

        ret_dict['label'] = label

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
    # df = df[df['study_id'] == 2780132468]

    aux_info = get_train_study_aux_info(data_root)

    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    create_dir(debug_dir)
    import albumentations as A

    AUG_PROB = 1.0
    ###
    transforms_train = A.Compose([
        # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=AUG_PROB),
        # A.OneOf([
        #     A.MotionBlur(blur_limit=5),
        #     A.MedianBlur(blur_limit=5),
        #     A.GaussianBlur(blur_limit=5),
        # ], p=AUG_PROB),

        #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=AUG_PROB),
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transforms_axial = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dset = RSNA24Dataset_Sag_Axial_Cls(data_root,
                                       aux_info,
                                       df,
                                       phase='valid',
                                       transform=transforms_train,
                                       axial_transform=transforms_axial,
                                       with_sag=True,
                                       with_axial=False,
                                       axial_z_imgs=3
                                       )
    print(len(dset))
    for d in dset:
        x = d['img']
        print('x: ', x.shape)
        cv2.imwrite(f'{debug_dir}/sag_t2_rbox.jpg', convert_to_cv2_img(x))
        break
