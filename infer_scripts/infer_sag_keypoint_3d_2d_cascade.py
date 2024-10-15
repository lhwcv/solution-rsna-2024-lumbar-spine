# -*- coding: utf-8 -*-
import os
import pickle

import timm_3d

os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import KFold

import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import timm
import random

import glob
import pydicom
from skimage.transform import resize
import warnings

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

G_IMG_SIZE = 512


def get_test_des_to_sids(data_root, test_series_descriptions_fn):
    test_series = pd.read_csv(
        f'{data_root}/{test_series_descriptions_fn}')

    infos = {}
    for study_id in test_series['study_id'].unique().tolist():
        study_series = test_series[test_series['study_id'] == study_id]
        des_to_sid = {}
        for sid, des in zip(study_series['series_id'], study_series['series_description']):
            if des in des_to_sid.keys():
                des_to_sid[des].append(sid)
            else:
                des_to_sid[des] = [sid]
        infos[study_id] = {
            'des_to_sid': des_to_sid
        }
    return infos


def convert_to_8bit_lhw_version(x):
    lower, upper = np.percentile(x, (1, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
    return x.astype("uint8")


def resize_volume(x, w=640, h=640):
    xs = []
    for i in range(len(x)):
        img = x[i]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        xs.append(img)
    return np.array(xs, dtype=np.uint8)


def load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    if len(dicom_files) == 0:
        return {
            'array': None
        }
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicom_instance_numbers = [int(i.split('/')[-1][:-4]) for i in dicom_files]

    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    dicom_instance_numbers = np.array(dicom_instance_numbers)[idx]

    cols = np.asarray([d.pixel_array.shape[1] for d in dicoms]).astype("int")[idx].tolist()
    rows = np.asarray([d.pixel_array.shape[0] for d in dicoms]).astype("int")[idx].tolist()
    instance_num_to_shape = {}
    dicom_instance_numbers = dicom_instance_numbers.tolist()
    for i, n in enumerate(dicom_instance_numbers):
        instance_num_to_shape[n] = [cols[i], rows[i]]

    array = []
    for i, d in enumerate(dicoms):
        arr = d.pixel_array.astype("float32")
        arr = resize(arr, (512, 512))
        array.append(arr)
    array = np.array(array)
    array = array[idx]
    array = convert_to_8bit_lhw_version(array)
    # array = resize_volume(array, 512, 512)

    return {"array": array, }


class RSNA24DatasetTest_LHW_keypoint_3D_Saggital(Dataset):
    def __init__(self,
                 data_root,
                 test_series_descriptions_fn,
                 study_ids,
                 transform=None,
                 image_dir=None):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root, test_series_descriptions_fn)

        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

    def __len__(self):
        return len(self.study_ids)

    def get_3d_volume(self, study_id, sid, series_description):
        # TODO use multi when test
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))
        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            plane = "axial"
            reverse_sort = True
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']
        return volume, sid

    def __getitem__(self, idx):

        study_id = self.study_ids[idx]
        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']

        # Sagittal T1
        has_t1 = True
        if 'Sagittal T1' not in des_to_sid.keys():
            has_t1 = False
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
        else:
            # TODO use multi
            sid = des_to_sid['Sagittal T1'][0]
            s_t1, _ = self.get_3d_volume(study_id, sid, "Sagittal T1")

        # Sagittal T2/STIR
        has_t2 = True
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            has_t2 = False
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)
        else:
            # TODO use multi
            sid = des_to_sid['Sagittal T2/STIR'][0]
            s_t2, _ = self.get_3d_volume(study_id, sid, "Sagittal T2/STIR")

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t1 = self.transform(image=s_t1)['image']
            s_t1 = s_t1.transpose(2, 0, 1)

            s_t2 = s_t2.transpose(1, 2, 0)
            s_t2 = self.transform(image=s_t2)['image']
            s_t2 = s_t2.transpose(2, 0, 1)

        s_t1 = s_t1.astype(np.float32)
        s_t2 = s_t2.astype(np.float32)

        s_t1_3d = torch.from_numpy(s_t1).unsqueeze(0).unsqueeze(0)
        s_t1_3d = F.interpolate(s_t1_3d, size=(48, 256, 256)).squeeze(0)
        s_t2_3d = torch.from_numpy(s_t2).unsqueeze(0).unsqueeze(0)
        s_t2_3d = F.interpolate(s_t2_3d, size=(48, 256, 256)).squeeze(0)
        x = torch.cat((s_t1_3d, s_t2_3d), dim=0)

        return {
            'x': x,
            's_t1': torch.from_numpy(s_t1),
            's_t2': torch.from_numpy(s_t2),
            'c1': len(s_t1),
            'c2': len(s_t2),
            'study_id': study_id
        }


autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)


class RSNA24Model_Keypoint_2D(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=1, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=10,
            global_pool='avg'
        )

    def forward(self, x):
        x = self.model(x)
        return x


class RSNA24Model_Keypoint_3D_Sag(nn.Module):
    def __init__(self, model_name='densenet161', pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=2,
            num_classes=45,
            global_pool='avg'
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x


def infer_keypoint_3d_2d_cascade(x, s_t1, s_t2, c1, c2, models_3d,
                                 models_2d_t1, models_2d_t2):
    bs = x.shape[0]
    with autocast:
        keypoints_3d = None
        for i in range(len(models_3d)):
            y = models_3d[i](x)

            if keypoints_3d is None:
                keypoints_3d = y.cpu().reshape(bs, 15, 3)
            else:
                keypoints_3d += y.cpu().reshape(bs, 15, 3)
        keypoints_3d = keypoints_3d / len(models_3d)

    keypoints_3d_origin = keypoints_3d.numpy()
    #
    # assemble 2d input by pred z
    keypoints_3d[:, :, :2] = 512 * keypoints_3d[:, :, :2] / 4.0
    keypoints_3d[:, 5, 2] = c2.reshape(-1, 1, 1) * keypoints_3d[:, 5, 2] / 16.0
    keypoints_3d[:, 5:, 2] = c1.reshape(-1, 1, 1) * keypoints_3d[:, 5:, 2] / 16.0
    keypoints_3d = keypoints_3d.numpy()
    keypoints_z = keypoints_3d[:, :, 2]
    keypoints_z = np.round(keypoints_z).astype(np.int64)

    t2_z = keypoints_z[:, :5]

    # print('t2_z: ', t2_z)
    t2_imgs = []
    for b in range(bs):
        for i in range(5):
            z = t2_z[b, i]
            if z < 0:
                z = 0
            if z > c2[b] - 1:
                z = c2[b] - 1
            t2_z[b, i] = z
            img = s_t2[b, z].unsqueeze(0).unsqueeze(0)
            t2_imgs.append(img)
    t2_imgs = torch.cat(t2_imgs, dim=0)  # bs*5, 1, 512, 512

    # t2_keypoints = torch.LongTensor(t2_z).to(s_t2.device)
    # t2_img_list = [torch.index_select(s_t2[b], dim=0, index=t2_keypoints[b]) for b in range(bs)]
    # t2_imgs = torch.cat(t2_img_list, dim=0).unsqueeze(1)

    t2_keypoints_xy = None

    with autocast:
        for i in range(len(models_2d_t2)):
            # x_flip = torch.flip(t2_imgs, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
            # p1 = models_2d_t2[i](t2_imgs)
            # p2 = models_2d_t2[i](x_flip)
            # p = (p1 + p2) / 2
            p = models_2d_t2[i](t2_imgs)
            if t2_keypoints_xy is None:
                t2_keypoints_xy = p
            else:
                t2_keypoints_xy += p

    t2_keypoints_xy = t2_keypoints_xy / len(models_2d_t2)
    #
    # t2_keypoints_xy = 512 * t2_keypoints_xy.cpu().reshape(bs, 5, 5, 2) / 4.0
    t2_keypoints_xy = 512 * t2_keypoints_xy.cpu().reshape(bs, 5, 5, 2) / 4.0
    t2_z = torch.from_numpy(t2_z)
    t2_z = t2_z.reshape(bs, 5, 1, 1).repeat((1, 1, 5, 1))
    t2_keypoints = torch.cat((t2_keypoints_xy, t2_z), dim=-1).numpy()

    ## t1 keypoints
    t1_z = keypoints_z[:, 5:]

    # print('t1_z: ', t1_z)
    t1_imgs = []
    for b in range(bs):
        for i in range(10):
            z = t1_z[b, i]
            if z < 0:
                z = 0
            if z > c1[b] - 1:
                z = c1[b] - 1
            t1_z[b, i] = z
            img = s_t1[b, z].unsqueeze(0).unsqueeze(0)
            t1_imgs.append(img)
    t1_imgs = torch.cat(t1_imgs, dim=0)  # bs*10, 1, 512, 512

    # t1_keypoints = torch.LongTensor(t1_z).to(s_t1.device)
    # t1_img_list = [torch.index_select(s_t1[b], dim=0, index=t1_keypoints[b]) for b in range(bs)]
    # t1_imgs = torch.cat(t1_img_list, dim=0).unsqueeze(1)

    t1_keypoints_xy = None

    with autocast:
        for i in range(len(models_2d_t1)):
            if t1_keypoints_xy is None:
                t1_keypoints_xy = models_2d_t1[i](t1_imgs)
            else:
                t1_keypoints_xy += models_2d_t1[i](t1_imgs)
    t1_keypoints_xy = t1_keypoints_xy / len(models_2d_t1)
    #
    t1_keypoints_xy = 512 * t1_keypoints_xy.cpu().reshape(bs, 10, 5, 2) / 4.0
    t1_z = torch.from_numpy(t1_z)
    t1_z = t1_z.reshape(bs, 10, 1, 1).repeat((1, 1, 5, 1))
    t1_keypoints = torch.cat((t1_keypoints_xy, t1_z), dim=-1).numpy()

    return t2_keypoints, t1_keypoints, keypoints_3d_origin


if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train_series_descriptions.csv')
    # large_err_list = [324739602, 859100583, 1510451897, 1880970480, 1901348744, 3335451812, 38281420, 2151467507, 2151509334, 3713534743, 4201106871, 286903519, 2316015842, 2444340715]
    # df = df[df['study_id'].isin(large_err_list)]

    study_ids = list(df['study_id'].unique())
    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    create_dir(debug_dir)

    models_3d_fns = [
        '../train_scripts/wkdir/keypoint/sag_3d_256/densenet161_lr_0.0006/best_fold_0_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_3d_256/densenet161_lr_0.0006/best_fold_1_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_3d_256/densenet161_lr_0.0006/best_fold_2_ema.pt'
    ]
    models_2d_t1_fns = [
        '../train_scripts/wkdir/keypoint/sag_t1_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_0_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_t1_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_1_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_t1_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_2_ema.pt'
    ]
    models_2d_t2_fns = [
        '../train_scripts/wkdir/keypoint/sag_t2_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_0_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_t2_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_1_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_t2_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_2_ema.pt',
        # '../train_scripts/wkdir/keypoint/sag_t2_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_3_ema.pt',
        # '../train_scripts/wkdir/keypoint/sag_t2_2d_5fold_densenet161/densenet161_lr_0.0006/best_fold_4_ema.pt',
    ]
    models_3d = []
    models_2d_t1, models_2d_t2 = [], []
    for fn in models_3d_fns:
        print('load: ', fn)
        model = RSNA24Model_Keypoint_3D_Sag('densenet161', pretrained=False).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models_3d.append(model)

    for fn in models_2d_t1_fns:
        print('load: ', fn)
        model = RSNA24Model_Keypoint_2D('densenet161', pretrained=False).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models_2d_t1.append(model)

    for fn in models_2d_t2_fns:
        print('load: ', fn)
        model = RSNA24Model_Keypoint_2D('densenet161', pretrained=False).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models_2d_t2.append(model)

    dset = RSNA24DatasetTest_LHW_keypoint_3D_Saggital(data_root,
                                                      'train_series_descriptions.csv',
                                                      study_ids,
                                                      image_dir=data_root + '/train_images/',
                                                      )

    dloader = DataLoader(dset, batch_size=1, num_workers=8)

    study_id_to_pred_keypoints = {}
    for d in tqdm.tqdm(dloader):
        with torch.no_grad():
            x = d['x'].cuda()
            s_t1 = d['s_t1'].cuda()
            s_t2 = d['s_t2'].cuda()
            c1 = d['c1']
            c2 = d['c2']
            study_ids = d['study_id']

            t2_keypoints, t1_keypoints, keypoints_3d_origin = infer_keypoint_3d_2d_cascade(x, s_t1, s_t2,
                                                                                           c1, c2, models_3d,
                                                                                           models_2d_t1, models_2d_t2)
            if False:
                # t2_keypoints: b, 5, 5, 3
                print('t2_keypoints: ', t2_keypoints)
                imgs = s_t2[0].cpu().numpy()
                sub_imgs = []
                for i in range(5):
                    ps = t2_keypoints[0][i]
                    z = ps[0][2]  # share 1 z
                    img = imgs[int(z)]
                    img = 255 * (img - img.min()) / (img.max() - img.min())
                    img = np.array(img, dtype=np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    for p in ps:
                        x, y = int(p[0]), int(p[1])
                        # print(x, y)
                        img = cv2.circle(img, (x, y), 5, (0, 0, 255), 7)
                    sub_imgs.append(img)
                img_concat = np.concatenate(sub_imgs, axis=0)
                cv2.imwrite(f'{debug_dir}/pred/{study_ids[0]}_001_x_t2.jpg', img_concat)

                # t1_keypoints: b, 10, 5, 3
                print('t1_keypoints: ', t1_keypoints)
                imgs = s_t1[0].cpu().numpy()
                sub_imgs = []
                for i in range(10):
                    ps = t1_keypoints[0][i]
                    z = ps[0][2]  # share 1 z
                    img = imgs[int(z)]
                    img = 255 * (img - img.min()) / (img.max() - img.min())
                    img = np.array(img, dtype=np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    for p in ps:
                        x, y = int(p[0]), int(p[1])
                        # print(x, y)
                        img = cv2.circle(img, (x, y), 5, (0, 0, 255), 7)
                    sub_imgs.append(img)
                img_concat = np.concatenate(sub_imgs, axis=0)
                cv2.imwrite(f'{debug_dir}/pred/{study_ids[0]}_001_x_t1.jpg', img_concat)
                continue
                #exit(0)

            for idx, study_id in enumerate(study_ids):
                study_id = int(study_id)
                if study_id not in study_id_to_pred_keypoints.keys():
                    study_id_to_pred_keypoints[study_id] = {}

                #study_id_to_pred_keypoints[study_id]['t2_keypoints'] = t2_keypoints[idx]
                #study_id_to_pred_keypoints[study_id]['t1_keypoints'] = t1_keypoints[idx]

                p = keypoints_3d_origin[idx].reshape(15, 3)

                t2_p = t2_keypoints[idx]  # 5, 5, 3
                t1_p = t1_keypoints[idx]  # 10, 5, 3

                t2_p[:, :, :2] = 4 * t2_p[:, :, :2] / 512
                t1_p[:, :, :2] = 4 * t1_p[:, :, :2] / 512

                for i in range(5):
                    p[i][:2] = t2_p[i, i][:2]
                for i in range(5):
                    p[i + 5][:2] = t1_p[i, i][:2]
                for i in range(5):
                    p[i + 10][:2] = t1_p[i + 5, i][:2]

                study_id_to_pred_keypoints[study_id]['points'] = p



    pickle.dump(study_id_to_pred_keypoints,
                open(data_root + f'/v2_keypoint_3d_2d_cascade_0823.pkl', 'wb'))
