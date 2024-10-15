# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import glob
import pydicom
from skimage.transform import resize
import albumentations as A
import torch.nn.functional as F
from math import sin, cos
import errno
import timm
import timm_3d

autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def convert_to_8bit_lhw_version(x):
    lower, upper = np.percentile(x, (1, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
    return x.astype("uint8")


def load_dicom(dicom_folder,
               plane='axial',
               reverse_sort=True,
               img_size=512):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    if len(dicom_files) == 0:
        return None, None
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicom_instance_numbers = [int(i.split('/')[-1][:-4]) for i in dicom_files]

    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)

    # PatientPosition = dicoms[0].PatientPosition
    # dicom_instance_numbers = np.array(dicom_instance_numbers)
    # reverse_sort = False
    # if PatientPosition in ['FFS', 'FFP']:
    #     reverse_sort = True
    # idx = np.argsort(-dicom_instance_numbers if reverse_sort else dicom_instance_numbers)

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
        arr = resize(arr, (img_size, img_size))
        array.append(arr)
    array = np.array(array)
    array = array[idx]
    array = convert_to_8bit_lhw_version(array)

    # PixelSpacing = np.asarray([d.PixelSpacing for d in dicoms]).astype("float")[idx]
    # SliceThickness = np.asarray([d.SliceThickness for d in dicoms]).astype("float")[idx]
    # SpacingBetweenSlices = np.asarray([d.SpacingBetweenSlices for d in dicoms]).astype("float")[idx]
    # SliceLocation = np.asarray([d.SliceLocation for d in dicoms]).astype("float")[idx]

    dicom_instance_numbers_to_idx = {}
    for i, ins in enumerate(dicom_instance_numbers):
        dicom_instance_numbers_to_idx[ins] = i
    meta = {
        "ImagePositionPatient": ipp,
        # 'SliceLocation': SliceLocation,
        # "PixelSpacing": PixelSpacing,
        # "SliceThickness": SliceThickness,
        # "SpacingBetweenSlices": SpacingBetweenSlices,
        # "PatientPosition": PatientPosition,

        "instance_num_to_shape": instance_num_to_shape,
        "dicom_instance_numbers": dicom_instance_numbers,
        "dicom_instance_numbers_to_idx": dicom_instance_numbers_to_idx,
    }
    try:
        SpacingBetweenSlices = np.asarray([d.SpacingBetweenSlices for d in dicoms]).astype("float")[idx]
        meta['SpacingBetweenSlices'] = SpacingBetweenSlices
    except:
        print('[WARN] no SpacingBetweenSlices property in dicom')
        pass
    return array, meta


class Axial_Level_Dataset_Multi_V24(Dataset):
    def __init__(self,
                 data_dir,
                 study_ids,
                 transform=None,
                 test_series_descriptions_fn='test_series_descriptions.csv',
                 image_dir='test_images'):
        super(Axial_Level_Dataset_Multi_V24, self).__init__()
        desc_df = pd.read_csv(f"{data_dir}/{test_series_descriptions_fn}")
        self.image_dir = f"{data_dir}/{image_dir}/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]
        self.study_ids = []
        for study_id in study_ids:
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            if len(series_id_list) > 0:
                self.study_ids.append(study_id)
            else:
                print(f'[WARN] {study_id} has no Axial T2')

        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])

        self.transform = transform

    def __len__(self):
        return len(self.study_ids)

    def get_one_series(self, study_id, series_id,
                       base_size=512,
                       crop_size=384,
                       img_size=224):

        dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
        arr, meta = load_dicom(dicom_dir, img_size=base_size)

        if arr is None:
            print('[WARN Axial_Level_Dataset_Multi_V24] arr is none  ')
            arr = np.zeros((10, base_size, base_size), dtype=np.uint8)

        arr_origin = arr.copy()

        if crop_size != base_size:
            arr = arr.transpose(1, 2, 0)
            arr = A.center_crop(arr, crop_size, crop_size)
            arr = arr.transpose(2, 0, 1)

        if meta is not None:
            target_spacing = 3.0
            # IPP_z = meta['ImagePositionPatient'][:, 2] # not used now
            IPP_z = np.zeros((10,), dtype=np.float32)
            if target_spacing is not None:
                if 'SpacingBetweenSlices' in meta:
                    SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]
                    z_scale = SpacingBetweenSlices / target_spacing
                    depth = int(arr.shape[0] * z_scale)
                    if depth < 1:
                        depth = 1

                    # not used now
                    # v = torch.from_numpy(IPP_z).unsqueeze(0).unsqueeze(0)
                    # v = F.interpolate(v, size=(depth), mode='linear', align_corners=False)
                    # IPP_z = v.squeeze().numpy()

                    # target_size = (depth, img_size, img_size)
                    target_size = (depth, img_size, img_size)
                    arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                        size=target_size)[0][0].numpy()
                else:
                    depth = arr.shape[0]
                    target_size = (depth, img_size, img_size)
                    arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                        size=target_size)[0][0].numpy()

            Z_MIN = -794.237000
            Z_MAX = 351.35
            IPP_z = (IPP_z - Z_MIN) / (Z_MAX - Z_MIN)
        else:
            depth = arr.shape[0]
            target_size = (depth, img_size, img_size)
            arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                size=target_size).numpy().squeeze()
            IPP_z = np.zeros((10,), dtype=np.float32)

        if self.transform is not None:
            arr = arr.transpose(1, 2, 0)
            augmented = self.transform(image=arr)
            arr = augmented['image']
            arr = arr.transpose(2, 0, 1)

        ret = {"arr": arr,
               "arr_origin": arr_origin,
               "IPP_z": IPP_z,
               "IPP_z0": IPP_z[0],
               }

        return ret

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        g = self.desc_df[self.desc_df['study_id'] == study_id]
        series_id_list = g['series_id'].to_list()
        data_list = []

        for sid in series_id_list:
            data_d = self.get_one_series(study_id, sid)
            data_d['sid'] = sid
            data_list.append(data_d)

        data_list = sorted(data_list, key=lambda x: x['IPP_z0'], reverse=True)
        series_id_list = [d['sid'] for d in data_list]
        data_dict = {}
        keys = ['arr', 'IPP_z']
        for key in keys:
            v_list = []
            for i in range(len(data_list)):
                v_list.append(data_list[i][key])
            v = np.concatenate(v_list, axis=0)
            if key in ['label']:
                data_dict[key] = torch.from_numpy(v).long()
            else:
                data_dict[key] = torch.from_numpy(v).float()
        data_dict["study_id"] = study_id

        index_info = []
        start = 0
        coords = []
        for i in range(len(data_list)):
            z_len = data_list[i]['arr'].shape[0]
            index_info.append((start, start + z_len))
            start = z_len
            coords.append(np.arange(0, z_len))

        coords = np.concatenate(coords, axis=0)
        data_dict['index_info'] = index_info
        data_dict['coords'] = torch.from_numpy(coords).float()
        data_dict['series_id_list'] = series_id_list

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


def axial_v24_collate_fn(batch):
    return batch


def axial_v24_data_to_cuda(d, device=None):
    bs = len(d)
    for b in range(bs):
        for k in d[b].keys():
            if k not in ['study_id', 'index_info', 'series_id_list']:
                if isinstance(d[b][k], list):
                    for n in range(len(d[b][k])):
                        if device is None:
                            d[b][k][n] = d[b][k][n].cuda()
                        else:
                            d[b][k][n] = d[b][k][n].to(device)
                else:
                    if device is None:
                        d[b][k] = d[b][k].cuda()
                    else:
                        d[b][k] = d[b][k].to(device)
    return d


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
    # volume_crop = volume_crop.astype(np.float32)
    return volume_crop, None


class Sag_T2_Dataset_V24(Dataset):
    def __init__(self, data_dir,
                 study_ids,
                 pred_sag_keypoints_infos_3d,
                 transform=None,
                 test_series_descriptions_fn='test_series_descriptions.csv',
                 image_dir='test_images',
                 z_imgs=3,
                 img_size=512,
                 crop_size_h=64,
                 crop_size_w=128,
                 resize_to_size=128,
                 use_affine=False,
                 cache_dir=None,
                 other_crop_size_list=None,
                 ):
        super(Sag_T2_Dataset_V24, self).__init__()
        if cache_dir is None:
            cache_dir = data_dir + '/cache_sag/'
        self.img_size = img_size
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.resize_to_size = resize_to_size

        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/{test_series_descriptions_fn}")
        self.image_dir = f"{data_dir}/{image_dir}"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Sagittal T2/STIR'])]
        self.z_imgs = z_imgs
        print('[Sag_T2_Dataset] z_imgs: ', z_imgs)
        self.study_ids = study_ids
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])

        self.transform = transform
        self.cache_dir = cache_dir
        create_dir(self.cache_dir)
        self.pred_sag_keypoints_infos_3d = pred_sag_keypoints_infos_3d

        self.study_id_to_sids = {}
        for study_id in study_ids:
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            self.study_id_to_sids[study_id] = series_id_list

        self.other_crop_size_list = other_crop_size_list

    def __len__(self):
        return len(self.study_ids)

    def load_dicom_and_pts(self, study_id, series_id, img_size=512):
        fn1 = f'{self.cache_dir}/sag_t2_{study_id}_{series_id}_img.npz'
        if os.path.exists(fn1):
            arr = np.load(fn1)['arr_0']
            return arr
        else:
            dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=img_size, )
            if arr is None:
                arr = np.zeros((10, img_size, img_size), dtype=np.uint8)

            # np.savez_compressed(fn1, arr)
            return arr

    def get_item_by_study_id(self, study_id):

        sids = self.study_id_to_sids[study_id]

        img_size = self.img_size
        crop_size_h = self.crop_size_h
        crop_size_w = self.crop_size_w
        resize_to_size = self.resize_to_size
        ret_dict = {
            'study_id': study_id,
        }
        cond = [0] * 5 + [1] * 5 + [2] * 5
        cond = np.array(cond, dtype=np.int64)
        ret_dict['cond'] = cond
        if len(sids) == 0:
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            if self.z_imgs is not None:
                s_t2 = np.zeros((5, self.z_imgs, resize_to_size, resize_to_size), dtype=np.float32)
            else:
                s_t2 = np.zeros((5, 10, resize_to_size, resize_to_size), dtype=np.float32)

            if self.other_crop_size_list is not None:
                data_key = f'{self.z_imgs}_{crop_size_h}_{crop_size_w}'
                ret_dict[data_key] = s_t2

                for (z_imgs, crop_h, crop_w) in self.other_crop_size_list:
                    data_key = f'{z_imgs}_{crop_h}_{crop_w}'
                    s_t2 = np.zeros((5, z_imgs, resize_to_size, resize_to_size), dtype=np.float32)
                    ret_dict[data_key] = s_t2
            else:
                ret_dict['s_t2'] = s_t2
            return ret_dict
        else:
            # TODO use multi
            sid = sids[0]
            s_t2 = self.load_dicom_and_pts(study_id, sid, img_size)
            if self.z_imgs is not None:
                if s_t2.shape[0] < self.z_imgs:
                    print('[WARN]  s_t2.shape[0] < self.z_imgs')
                    d, h, w = s_t2.shape
                    s_t2 = torch.from_numpy(s_t2).unsqueeze(0).unsqueeze(1)
                    s_t2 = F.interpolate(s_t2, size=(self.z_imgs, h, w)).squeeze().numpy()

            c2 = s_t2.shape[0]
            pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id][sid].reshape(5, 3)
            pred_keypoints[:, :2] = img_size * pred_keypoints[:, :2]
            pred_keypoints[:, 2] = c2 * pred_keypoints[:, 2]
            pred_keypoints = np.round(pred_keypoints).astype(np.int64)
            t2_keypoints = pred_keypoints[:5]

            keypoints_xy = t2_keypoints[:, :2]
            keypoints_z = t2_keypoints[:, 2:]
            mask_xy = np.where(keypoints_xy < 0, 0, 1)
            mask_z = np.where(keypoints_z < 0, 0, 1)

            if self.transform is not None:
                # to h,w,c
                s_t2 = s_t2.transpose(1, 2, 0)
                augmented = self.transform(image=s_t2)
                s_t2 = augmented['image']
                s_t2 = s_t2.transpose(2, 0, 1)

            keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)
            mask = np.concatenate((mask_xy, mask_z), axis=1)
            s_t2_keypoints = keypoints

            s_t2_crop, _ = crop_out_by_keypoints(s_t2, s_t2_keypoints,
                                                 z_imgs=self.z_imgs,
                                                 crop_size_h=crop_size_h,
                                                 crop_size_w=crop_size_w,
                                                 resize_to_size=resize_to_size,
                                                 G_IMG_SIZE=img_size)

            if self.other_crop_size_list is not None:
                data_key = f'{self.z_imgs}_{crop_size_h}_{crop_size_w}'
                ret_dict[data_key] = s_t2_crop

                for (z_imgs, crop_h, crop_w) in self.other_crop_size_list:
                    data_key = f'{z_imgs}_{crop_h}_{crop_w}'
                    other_crop, _ = crop_out_by_keypoints(s_t2, s_t2_keypoints,
                                                          z_imgs=z_imgs,
                                                          crop_size_h=crop_h,
                                                          crop_size_w=crop_w,
                                                          resize_to_size=resize_to_size,
                                                          G_IMG_SIZE=img_size)

                    ret_dict[data_key] = other_crop
            else:
                ret_dict['s_t2'] = s_t2_crop

            return ret_dict

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        return self.get_item_by_study_id(study_id)


class Sag_T1_Dataset_V24(Dataset):
    def __init__(self, data_dir,
                 study_ids,
                 pred_sag_keypoints_infos_3d,
                 transform=None,
                 test_series_descriptions_fn='test_series_descriptions.csv',
                 image_dir='test_images',
                 z_imgs=3,
                 img_size=512,
                 crop_size_h=64,
                 crop_size_w=128,
                 resize_to_size=128,
                 use_affine=False,
                 cache_dir=None,
                 other_crop_size_list=None,
                 ):
        super(Sag_T1_Dataset_V24, self).__init__()
        if cache_dir is None:
            cache_dir = data_dir + '/cache_sag/'
        self.img_size = img_size
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.resize_to_size = resize_to_size

        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/{test_series_descriptions_fn}")
        self.image_dir = f"{data_dir}/{image_dir}"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Sagittal T1'])]
        self.z_imgs = z_imgs
        print('[Sag_T1_Dataset] z_imgs: ', z_imgs)
        self.study_ids = study_ids
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])

        self.transform = transform
        self.cache_dir = cache_dir
        create_dir(self.cache_dir)
        self.pred_sag_keypoints_infos_3d = pred_sag_keypoints_infos_3d

        self.study_id_to_sids = {}
        for study_id in study_ids:
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            if len(series_id_list) == 0:
                print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            self.study_id_to_sids[study_id] = series_id_list
        self.other_crop_size_list = other_crop_size_list

    def __len__(self):
        return len(self.study_ids)

    def load_dicom_and_pts(self, study_id, series_id, img_size=512):
        fn1 = f'{self.cache_dir}/sag_t1_{study_id}_{series_id}_img.npz'
        if os.path.exists(fn1):
            arr = np.load(fn1)['arr_0']
            return arr
        else:
            print('[warn] load without cache')
            dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=img_size, )
            if arr is None:
                arr = np.zeros((10, img_size, img_size), dtype=np.uint8)

            # np.savez_compressed(fn1, arr)
            return arr

    def get_item_by_study_id(self, study_id):

        sids = self.study_id_to_sids[study_id]
        img_size = self.img_size
        crop_size_h = self.crop_size_h
        crop_size_w = self.crop_size_w
        resize_to_size = self.resize_to_size
        ret_dict = {
            'study_id': study_id,
        }
        cond = [1] * 5 + [2] * 5
        cond = np.array(cond, dtype=np.int64)
        ret_dict['cond'] = cond

        if len(sids) == 0:
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            if self.z_imgs is not None:
                s_t1 = np.zeros((10, self.z_imgs, resize_to_size, resize_to_size), dtype=np.float32)
            else:
                s_t1 = np.zeros((10, 10, resize_to_size, resize_to_size), dtype=np.float32)

            if self.other_crop_size_list is not None:
                data_key = f'{self.z_imgs}_{crop_size_h}_{crop_size_w}'
                ret_dict[data_key] = s_t1

                for (z_imgs, crop_h, crop_w) in self.other_crop_size_list:
                    s_t1 = np.zeros((10, z_imgs, resize_to_size, resize_to_size), dtype=np.float32)
                    data_key = f'{z_imgs}_{crop_h}_{crop_w}'
                    ret_dict[data_key] = s_t1
            else:
                ret_dict['s_t1'] = s_t1

            return ret_dict
        else:
            # TODO use multi
            sid = sids[0]
            s_t1 = self.load_dicom_and_pts(study_id, sid, img_size)

            if self.z_imgs is not None:
                if s_t1.shape[0] < self.z_imgs:
                    print('[WARN]  s_t2.shape[0] < self.z_imgs')
                    d, h, w = s_t1.shape
                    s_t1 = torch.from_numpy(s_t1).unsqueeze(0).unsqueeze(1)
                    s_t1 = F.interpolate(s_t1, size=(self.z_imgs, h, w)).squeeze().numpy()

            c1 = s_t1.shape[0]
            pred_keypoints = self.pred_sag_keypoints_infos_3d[study_id][sid].reshape(10, 3)
            pred_keypoints[:, :2] = img_size * pred_keypoints[:, :2]
            pred_keypoints[:, 2] = c1 * pred_keypoints[:, 2]

            pred_keypoints = np.round(pred_keypoints).astype(np.int64)

            t1_keypoints = pred_keypoints
            keypoints_xy = t1_keypoints[:, :2]
            keypoints_z = t1_keypoints[:, 2:]

            if self.transform is not None:
                # to h,w,c
                s_t1 = s_t1.transpose(1, 2, 0)
                augmented = self.transform(image=s_t1)
                s_t1 = augmented['image']
                s_t1 = s_t1.transpose(2, 0, 1)

            keypoints = np.concatenate((keypoints_xy, keypoints_z), axis=1)

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
            s_t1_crop = np.concatenate([s_t1_left, s_t1_right], axis=0)

            if self.other_crop_size_list is not None:
                data_key = f'{self.z_imgs}_{crop_size_h}_{crop_size_w}'
                ret_dict[data_key] = s_t1_crop

                for (z_imgs, crop_h, crop_w) in self.other_crop_size_list:
                    data_key = f'{z_imgs}_{crop_h}_{crop_w}'
                    s_t1_left, _ = crop_out_by_keypoints(s_t1, s_t1_left_keypoints,
                                                         z_imgs=z_imgs,
                                                         crop_size_h=crop_h,
                                                         crop_size_w=crop_w,
                                                         resize_to_size=resize_to_size,
                                                         G_IMG_SIZE=img_size)
                    #
                    s_t1_right, _ = crop_out_by_keypoints(s_t1, s_t1_right_keypoints,
                                                          z_imgs=z_imgs,
                                                          crop_size_h=crop_h,
                                                          crop_size_w=crop_w,
                                                          resize_to_size=resize_to_size,
                                                          G_IMG_SIZE=img_size)
                    other_crop = np.concatenate([s_t1_left, s_t1_right], axis=0)

                    ret_dict[data_key] = other_crop
            else:
                ret_dict['s_t1'] = s_t1_crop
            return ret_dict

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        return self.get_item_by_study_id(study_id)


class Sag_3D_Point_Dataset_V24(Dataset):
    def __init__(self,
                 data_dir,
                 study_ids,
                 transform=None,
                 test_series_descriptions_fn='test_series_descriptions.csv',
                 image_dir='test_images',
                 series_description='Sagittal T2/STIR',
                 img_size=512,
                 depth_3d=32,
                 img_size_3d=384,
                 cache_dir=None,
                 with_origin_arr=False,
                 ):
        super(Sag_3D_Point_Dataset_V24, self).__init__()
        if cache_dir is None:
            cache_dir = data_dir + '/cache_sag/'
        self.with_origin_arr = with_origin_arr
        self.depth_3d = depth_3d
        self.img_size_3d = img_size_3d
        self.data_dir = data_dir
        desc_df = pd.read_csv(f"{data_dir}/{test_series_descriptions_fn}")
        self.image_dir = f"{data_dir}/{image_dir}/"
        self.desc_df = desc_df[desc_df['series_description'].isin([series_description])]
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])

        self.transform = transform
        self.img_size = img_size

        self.study_ids = study_ids
        self.series_description = series_description

        self.cache_dir = cache_dir
        create_dir(self.cache_dir)

        self.samples = []
        for study_id in study_ids:
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            for series_id in series_id_list:
                self.samples.append({
                    'study_id': study_id,
                    'series_id': series_id,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        study_id = sample['study_id']
        series_id = sample['series_id']
        prefix = 'sag_t1'
        if self.series_description == 'Sagittal T2/STIR':
            prefix = 'sag_t2'
        fn1 = f'{self.cache_dir}/{prefix}_{study_id}_{series_id}_img.npz'

        if os.path.exists(fn1):
            arr = np.load(fn1)['arr_0']

        else:
            print('[warn] load with uncache, load maybe be slow?')
            dicom_dir = f'{self.data_dir}/test_images/{study_id}/{series_id}/'
            arr, meta = load_dicom(dicom_dir,
                                   plane='sagittal', reverse_sort=False,
                                   img_size=self.img_size, )
            if arr is None:
                arr = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)

            # np.savez_compressed(fn1, arr)

        if self.transform is not None:
            # to h,w,c
            arr = arr.transpose(1, 2, 0)
            augmented = self.transform(image=arr)
            arr = augmented['image']
            arr = arr.transpose(2, 0, 1)

        origin_depth = arr.shape[0]
        arr = arr.astype(np.float32)
        arr = torch.from_numpy(arr)

        imgs = arr.unsqueeze(0).unsqueeze(0)
        imgs = F.interpolate(imgs, size=(self.depth_3d, self.img_size_3d, self.img_size_3d)).squeeze(0)

        ret = {
            'imgs': imgs,
            # 'origin_imgs': arr,
            'study_id': study_id,
            'series_id': series_id,
            'origin_depth': origin_depth,
        }
        if self.with_origin_arr:
            ret['origin_imgs'] = arr
        return ret


class Axial_Cond_Dataset_Multi_V24(Dataset):
    def __init__(self,
                 data_dir,
                 study_ids,
                 axial_pred_keypoints_info,
                 pred_sag_keypoints_infos_3d,
                 transform=None,
                 test_series_descriptions_fn='test_series_descriptions.csv',
                 image_dir='test_images',
                 z_imgs=5,
                 with_sag=True,
                 sag_transform=None,
                 sag_with_3d=False,
                 other_z_imgs_list=None,
                 cache_dir=None
                 ):
        super(Axial_Cond_Dataset_Multi_V24, self).__init__()
        desc_df = pd.read_csv(f"{data_dir}/{test_series_descriptions_fn}")
        self.data_dir = data_dir
        self.image_dir = f"{data_dir}/{image_dir}/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]

        self.z_imgs = z_imgs
        print('[Axial_Cond_Dataset_Multi_V24] z_imgs: ', z_imgs)
        self.study_ids = study_ids
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])

        self.transform = transform
        self.axial_pred_keypoints_info = axial_pred_keypoints_info
        self.with_sag = with_sag
        self.sag_with_3d = sag_with_3d
        self.other_z_imgs_list = other_z_imgs_list
        self.with_3_channel = True
        if self.z_imgs == 3:
            self.with_3_channel = False

        print('with_sag: ', with_sag)
        if with_sag:
            self.sag_t2_dset = Sag_T2_Dataset_V24(
                data_dir,
                study_ids,
                pred_sag_keypoints_infos_3d,
                transform=sag_transform,
                test_series_descriptions_fn=test_series_descriptions_fn,
                image_dir=image_dir, z_imgs=None if sag_with_3d else 7,
                img_size=512,
                crop_size_h=64,
                crop_size_w=128,
                resize_to_size=128,
                cache_dir=cache_dir
            )

    def __len__(self):
        return len(self.study_ids)

    def get_one_series(self, study_id, series_id,
                       base_size=512,
                       crop_size=256,
                       img_size=256,
                       target_spacing=3.0):

        dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
        arr, meta = load_dicom(dicom_dir, img_size=base_size)
        if arr is None:
            arr = np.zeros((10, base_size, base_size), dtype=np.uint8)

        if crop_size != base_size:
            arr = arr.transpose(1, 2, 0)
            arr = A.center_crop(arr, crop_size, crop_size)
            arr = arr.transpose(2, 0, 1)

        if target_spacing != None:
            SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]
            z_scale = SpacingBetweenSlices / target_spacing
            depth = int(arr.shape[0] * z_scale)
            target_size = (depth, img_size, img_size)
            arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                                size=target_size).numpy().squeeze()

        return arr

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
                                             transform=None,
                                             with_3_channel=True
                                             ):
        imgs = []
        for level in range(5):
            if level not in level_to_imgs_dict.keys():
                if with_3_channel:
                    img = np.zeros((z_imgs, 3, image_size, image_size), dtype=np.float32)
                else:
                    img = np.zeros((z_imgs, 1, image_size, image_size), dtype=np.float32)
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
                img, score = level_to_imgs_dict[level][0].copy()
                if with_3_channel:
                    ind = np.arange(img.shape[0])
                    ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
                    ind_a = np.where(np.array(ind) + 1 >= img.shape[0] - 1, img.shape[0] - 1, np.array(ind) + 1)
                    img = np.stack([img[ind_b, :, :], img, img[ind_a, :, :]], axis=-1)

                if transform is not None:
                    img = [transform(image=i)['image'] for i in img]
                    img = np.array(img)
                    if with_3_channel:
                        img = img.transpose(0, 3, 1, 2)
                    # img = img.transpose(1, 2, 0)
                    # img = transform(image=img)['image']
                    # img = img.transpose(2, 0, 1)
                if not with_3_channel:
                    img = img[:, np.newaxis, :, :]
                imgs.append(img)
        imgs = np.array(imgs)  # 5, 5, 3, 256, 256

        # print('imgs shape: ', imgs.shape)
        # exit(0)
        imgs = np.array(imgs, dtype=np.float32)
        return imgs

    def get_pred_keypoints(self, study_id, sid, arr,
                           base_size=720, crop_size=512, img_size=512, ):
        pts = self.axial_pred_keypoints_info[study_id][sid]['points']  # 2, 5, 4
        pts = pts.transpose(1, 0, 2)  # 5, 2, 4
        pts = np.ascontiguousarray(pts)
        # print('pts: ',pts.shape)
        z_len, h, w = arr.shape

        origin_base_size = 512

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

    def crop_left_right_imgs(self,z_imgs, keypoints_list, arr_list, img_size,
                             roi_img_size = 128):
        level_to_imgs_left = {}
        level_to_imgs_right = {}
        # debug_imgs = []

        for i in range(len(keypoints_list)):
            keypoints = keypoints_list[i]
            arr = arr_list[i]
            # arr n, h, w
            # if n < 5, repeat to 5
            if arr.shape[0] < 5:
                repeat_times = 5 // arr.shape[0]
                arr = np.repeat(arr, repeat_times, axis=0)
                # 若重复后的维度大于5，截取
                if arr.shape[0] > 5:
                    arr = arr[:5, :, :]

            for level in range(5):
                left_p = keypoints[level, 0]
                if left_p[2] >= 0:
                    img, _ = crop_out_by_keypoints(arr,
                                                   [left_p],
                                                   z_imgs=z_imgs,
                                                   crop_size_w=roi_img_size,
                                                   crop_size_h=roi_img_size,
                                                   G_IMG_SIZE=img_size)
                    img = img[0]

                    if level in level_to_imgs_left.keys():
                        level_to_imgs_left[level].append([img, left_p[3]])
                    else:
                        level_to_imgs_left[level] = [[img, left_p[3]]]

                right_p = keypoints[level, 1]
                if right_p[2] >= 0:
                    img, _ = crop_out_by_keypoints(arr,
                                                   [right_p],
                                                   z_imgs=z_imgs,
                                                   crop_size_w=roi_img_size,
                                                   crop_size_h=roi_img_size,
                                                   G_IMG_SIZE=img_size)
                    img = img[0]

                    if level in level_to_imgs_right.keys():
                        level_to_imgs_right[level].append([img, right_p[3]])
                    else:
                        level_to_imgs_right[level] = [[img, right_p[3]]]
        return level_to_imgs_left, level_to_imgs_right

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]

        g = self.desc_df[self.desc_df['study_id'] == study_id]
        series_id_list = g['series_id'].to_list()
        arr_list = []
        keypoints_list = []
        base_size = 720
        crop_size = 512
        img_size = 512
        target_spacing = None

        for sid in series_id_list:
            arr = self.get_one_series(study_id, sid,
                                      base_size, crop_size,
                                      img_size, target_spacing)

            arr_list.append(arr)
            pred_keypoints = self.get_pred_keypoints(study_id, sid, arr,
                                                     base_size, crop_size, img_size, )
            pred_keypoints = np.round(pred_keypoints)
            keypoints_list.append(pred_keypoints)

        roi_img_size = 128
        level_to_imgs_left, level_to_imgs_right = self.crop_left_right_imgs(
            self.z_imgs, keypoints_list, arr_list, img_size, roi_img_size
        )
        left_imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs_left,
                                                              z_imgs=self.z_imgs,
                                                              image_size=roi_img_size,
                                                              transform=self.transform,
                                                              with_3_channel=self.with_3_channel)
        right_imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs_right,
                                                               z_imgs=self.z_imgs,
                                                               image_size=roi_img_size,
                                                               transform=self.transform,
                                                               with_3_channel=self.with_3_channel)
        right_imgs = np.flip(right_imgs, axis=[-1])
        axial_imgs = np.concatenate((left_imgs, right_imgs), axis=0)
        axial_imgs = axial_imgs.astype(np.float32)

        data_dict = {
            'study_id': study_id,
        }
        if self.other_z_imgs_list is not None:
            data_key = f'axial_imgs_{self.z_imgs}_{roi_img_size}_{roi_img_size}'
            data_dict[data_key] = axial_imgs
            for z_imgs, crop_h, crop_w in self.other_z_imgs_list:
                data_key = f'axial_imgs_{z_imgs}_{crop_h}_{crop_w}'
                with_3_channel = True
                if z_imgs <= 3:
                    with_3_channel = False

                level_to_imgs_left, level_to_imgs_right = self.crop_left_right_imgs(
                    z_imgs, keypoints_list, arr_list, img_size, roi_img_size
                )

                left_imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs_left,
                                                                      z_imgs=z_imgs,
                                                                      image_size=crop_h,
                                                                      transform=self.transform,
                                                                      with_3_channel=with_3_channel)
                right_imgs = self.assemble_5_level_imgs_with_transform(level_to_imgs_right,
                                                                       z_imgs=z_imgs,
                                                                       image_size=crop_h,
                                                                       transform=self.transform,
                                                                       with_3_channel=with_3_channel)
                right_imgs = np.flip(right_imgs, axis=[-1])
                axial_imgs = np.concatenate((left_imgs, right_imgs), axis=0)
                axial_imgs = axial_imgs.astype(np.float32)
                data_dict[data_key] = axial_imgs
        else:
            data_dict['axial_imgs'] = axial_imgs,

        if self.with_sag:
            sag_t2 = self.sag_t2_dset.get_item_by_study_id(study_id)['s_t2']
            # print('sag_t2: ', sag_t2.shape)
            if self.sag_with_3d:
                sag_t2 = torch.from_numpy(sag_t2).unsqueeze(1)
                sag_t2 = F.interpolate(sag_t2, size=(32, 96, 96)).squeeze().numpy()
            else:

                cs = sag_t2.shape[1] // 2
                left_side = sag_t2[:, cs:, :, :]
                # right_side = sag_t2[:, :cs, :, :]
                right_side = sag_t2[:, :cs + 1, :, :]
                right_side = np.flip(right_side, axis=[1])
                right_side = np.ascontiguousarray(right_side)
                sag_t2 = np.concatenate((left_side, right_side), axis=0)

            data_dict['sag_t2'] = sag_t2.astype(np.float32)

        return data_dict


### models
class Axial_Level_Cls_Encoder(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=3, pretrained=True):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=3,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.lstm = nn.LSTM(fea_dim,
                            fea_dim // 2,
                            bidirectional=True,
                            batch_first=True, num_layers=2)

        self.xy_reg = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 4),
        )

        self.classifier = nn.Linear(fea_dim, 5)

    def forward(self, x, return_fea=False):
        # x = F.interpolate(x, size=(160, 160))
        # b, z_len, 256, 256
        x = F.pad(x, (0, 0, 0, 0, 1, 1))
        # 使用 unfold 函数进行划窗操作
        x = x.unfold(1, 3, 1)  # bs, z_len, 256, 256, 3
        x = x.permute(0, 1, 4, 2, 3)  # bs, z_len, 3, 256, 256
        # x = x.unsqueeze(2)  # bs, z_len, 1, 256, 256
        bs, z_len, c, h, w = x.shape
        x = x.reshape(bs * z_len, c, h, w)
        x = self.model(x).reshape(bs, z_len, -1)

        xy_pred = self.xy_reg(x)  # bs, z_len, 4
        xy_pred = xy_pred.reshape(bs, z_len, 2, 2)
        # xy_pred = xy_pred.permute(0, 2, 1, 3)  # bs, 2, z_len, 2

        x0, _ = self.lstm(x)
        x = x + x0
        out = self.classifier(x)
        if return_fea:
            return out, xy_pred, x
        return out, xy_pred


class PositionalEncoding(nn.Module):
    def __init__(self, fea_dim=512):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(fea_dim, fea_dim)

    def forward(self, x, IPP_z):
        x = self.linear(x) + IPP_z.unsqueeze(-1)
        return x


class Axial_Level_Cls_Model_for_Test(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=True):
        super(Axial_Level_Cls_Model_for_Test, self).__init__()
        self.cnn_encoder = Axial_Level_Cls_Encoder(model_name, pretrained=pretrained)

        hidden_size = 512
        # self.pe = PositionalEncoding(fea_dim=hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=4,
                                                   dim_feedforward=hidden_size * 4,
                                                   dropout=0.1)
        self.ts_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.level_classifier = nn.Linear(hidden_size, 5)

        self.left_classifier = nn.Linear(hidden_size, 5)
        self.right_classifier = nn.Linear(hidden_size, 5)

    def forward_arr(self, arr, IPP_z):
        level_cls_pred, xy_pred, fea = self.cnn_encoder(arr, return_fea=True)
        # fea = self.pe(fea, IPP_z)
        fea = self.ts_encoder(fea)
        level_cls_pred = self.level_classifier(fea)
        # print('level_cls_pred: ', level_cls_pred.shape)
        left_pred = self.left_classifier(fea).unsqueeze(2)
        right_pred = self.right_classifier(fea).unsqueeze(2)
        sparse_pred = torch.cat((left_pred, right_pred), dim=2)
        return level_cls_pred, sparse_pred, xy_pred

    def forward(self, d):
        arr = d[0]['arr'].unsqueeze(0)
        IPP_z = d[0]['IPP_z'].unsqueeze(0)
        level_cls_pred, sparse_pred, xy_pred = self.forward_arr(arr, IPP_z)
        if not self.training:
            x_flip1 = torch.flip(arr, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
            x_flip2 = torch.flip(arr, dims=[-2, ])  # A.VerticalFlip(p=0.5),
            level_cls_pred1, sparse_pred1, xy_pred1 = self.forward_arr(x_flip1, IPP_z)
            level_cls_pred2, sparse_pred2, xy_pred2 = self.forward_arr(x_flip2, IPP_z)
            level_cls_pred = (level_cls_pred + level_cls_pred1 + level_cls_pred2) / 3
            sparse_pred = (sparse_pred + sparse_pred1 + sparse_pred2) / 3

        coords = d[0]['coords']
        index_info = d[0]['index_info']

        return {
            'level_cls_pred': level_cls_pred,
            'sparse_pred': sparse_pred,
            'coords': coords,
            'index_info': index_info,
        }


class RSNA24Model_Keypoint_2D(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=1, pretrained=False,
                 num_classes=10):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool='avg'
        )

    def forward(self, x):
        x = self.model(x)
        return x


class RSNA24Model_Keypoint_3D(nn.Module):
    def __init__(self, model_name='densenet161',
                 in_chans=1,
                 num_classes=30,
                 pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool='avg'
        )

    def forward(self, x):
        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x


class Sag_Model_25D_Level_LSTM(nn.Module):
    def __init__(self,
                 model_name='densenet201',
                 in_chans=3,
                 n_classes=3,
                 pretrained=False,
                 with_level_lstm=True,
                 with_emb=False
                 ):
        super().__init__()
        fea_dim = 512
        self.n_classes = n_classes
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.drop = nn.Dropout(0.1)
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, n_classes)
        )
        self.level_lstm = None
        if with_level_lstm:
            self.level_lstm = nn.LSTM(fea_dim,
                                      fea_dim // 2,
                                      bidirectional=True,
                                      batch_first=True, num_layers=2)
        self.out_fea_extractor = None
        if with_emb:
            self.out_fea_extractor = nn.Sequential(
                nn.Linear(fea_dim, fea_dim),
                nn.LeakyReLU(),
                nn.Linear(fea_dim, 128)
            )

    def forward_train(self, x):
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        n_conds = k // 5

        x = x.reshape(b * k, d, h, w)
        x = self.model(x)

        if self.level_lstm is not None:
            # bs, n_conditions, 5_level, fdim
            x = x.reshape(b * n_conds, 5, -1)

            # worse with sub mean
            # xm = x.mean(dim=1, keepdims=True)
            # x = x - xm
            xm, _ = self.level_lstm(x)
            x = x + xm
            x = x.reshape(b * k, -1)

        x = self.drop(x)
        if self.out_fea_extractor is not None:
            embeds = self.out_fea_extractor(x)
            # bs, n_cond, 5_level, 128
            embeds = embeds.reshape(b, -1, 5, 128)

        x = self.out_linear(x)
        x = x.reshape(b, k, self.n_classes)
        x = x.reshape(b, -1)
        if self.out_fea_extractor is not None:
            return x, embeds
        return x

    def forward(self, x):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x0 = self.forward_train(x)
        x1 = self.forward_train(x_flip)
        return (x0 + x1) / 2


class Sag_Model_25D_GRU(nn.Module):
    def __init__(self,
                 model_name='densenet201',
                 pretrained=True,
                 with_emb=False):
        super(Sag_Model_25D_GRU, self).__init__()

        self.model = timm.create_model(model_name,
                                       pretrained=pretrained,
                                       num_classes=0, in_chans=1)
        nc = self.model.num_features

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, 3)

        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_fea_extractor = None
        if with_emb:
            self.out_fea_extractor = nn.Sequential(
                nn.Linear(2048, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512)
            )

    def forward_train(self, x):
        # print('x shape: ', x.shape)
        bs, k, n, h, w = x.size()
        x = x.reshape(bs * k * n, 1, h, w)
        x = self.model(x)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 3) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 3)

        y = y.reshape(bs, -1, 5, 3)
        y = y.reshape(bs, -1)

        if self.out_fea_extractor is not None:
            embeds = embeds.reshape(bs, 3, 5, 1024)
            embeds = embeds.permute(0, 2, 1, 3).reshape(bs, 5, -1)
            embeds = self.out_fea_extractor(embeds)  # bs, 5, 512
            return y, embeds

        return y

    def forward(self, x):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x0 = self.forward_train(x)
        x1 = self.forward_train(x_flip)
        return (x0 + x1) / 2


def build_sag_model(model_name='densenet201',
                    in_chans=3,
                    n_classes=3,
                    pretrained=False,
                    with_level_lstm=True,
                    with_emb=False,
                    with_gru=False):
    if not with_gru:
        return Sag_Model_25D_Level_LSTM(model_name,
                                        in_chans,
                                        n_classes,
                                        pretrained,
                                        with_level_lstm,
                                        with_emb)
    else:
        return Sag_Model_25D_GRU(model_name, pretrained, with_emb)


class Axial_Model_25D(nn.Module):
    def __init__(self, model_name='densenet201', in_chans=3,
                 n_classes=4, pretrained=False):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.drop = nn.Dropout(0.1)
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 6)
        )

    def forward(self, x):
        # 5 level: b, 10, d, h, w
        b, k, d, h, w = x.shape

        n_conds = k // 5

        x = x.reshape(b * k, d, h, w)
        x = self.model(x)

        x = self.drop(x)

        x = self.out_linear(x)
        x = x.reshape(b, k, 6)
        left_x = x[:, :k // 2, :3]
        right_x = x[:, k // 2:, 3:]

        # b, k, 3
        x = torch.cat((left_x, right_x), dim=1)

        x = x.reshape(b, -1)
        return x


class Axial_Model_25D_GRU(nn.Module):
    def __init__(self, base_model='densenet201', axial_in_channels=3, pretrained=True):
        super(Axial_Model_25D_GRU, self).__init__()

        self.model = timm.create_model(base_model,
                                       pretrained=pretrained,
                                       num_classes=0, in_chans=axial_in_channels)
        nc = self.model.num_features

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, 3)

        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_fea_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, axial_imgs, with_emb=False):
        bs, k, n, c, h, w = axial_imgs.size()
        axial_imgs = axial_imgs.reshape(bs * k * n, -1, h, w)
        x = self.model(axial_imgs)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 3) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 3)

        y = y.reshape(bs, 2, 5, 3)
        y = y.reshape(bs, -1)

        if with_emb:
            # bs, 2_cond, 5_level, 1024
            embeds = embeds.reshape(bs, 2, 5, 1024)
            embeds = self.out_fea_extractor(embeds)  # bs, 2, 5, 128
            return y, embeds

        return y


class Axial_HybridModel_24(nn.Module):
    def __init__(self, backbone_sag='densenet201',
                 backbone_axial='densenet201',
                 pretrained=False,
                 axial_in_channels=3):
        super().__init__()
        self.sag_model = Sag_Model_25D_Level_LSTM(backbone_sag,
                                                  in_chans=4,
                                                  pretrained=pretrained,
                                                  with_emb=True,
                                                  with_level_lstm=True)
        self.axial_model = Axial_Model_25D_GRU(base_model=backbone_axial,
                                               pretrained=pretrained,
                                               axial_in_channels=axial_in_channels)
        fdim = 2 * 128
        self.out_linear = nn.Linear(fdim, 3)

    def forward(self, s_t2, axial_x):
        if self.training:
            return self.forward_train(s_t2, axial_x)
        else:
            return self.forward_test(s_t2, axial_x)

    def forward_train(self, s_t2, axial_x):
        _, sag_emb = self.sag_model.forward_train(s_t2)
        bs = axial_x.shape[0]

        axial_pred, axial_emb = self.axial_model.forward(axial_x, with_emb=True)
        # bs, 2_cond, 5_level, 3
        axial_pred = axial_pred.reshape(bs, 2, 5, 3)

        ### fuse ####
        # bs, 2_cond, 5_level, fdim
        fea = torch.cat((sag_emb, axial_emb), dim=-1)
        # bs, 2_cond, 5_level, 3
        ys = self.out_linear(fea)  # + axial_pred
        ys = ys.reshape(bs, -1)
        return ys

    def forward_test(self, x, axial_x):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),
        ys1 = self.forward_train(x, axial_x)
        ys2 = self.forward_train(x_flip, axial_x_flip)
        ys = (ys1 + ys2) / 2
        return ys


### infer


def axial_v24_infer_z(models, data_dict, device=None):
    data_dict = axial_v24_data_to_cuda(data_dict, device)

    sparse_pred = None
    coords = data_dict[0]['coords'].cpu()
    index_info = data_dict[0]['index_info']
    study_id = data_dict[0]['study_id']
    with torch.no_grad():
        with autocast:
            for i in range(len(models)):
                p = models[i](data_dict)
                if sparse_pred is None:
                    sparse_pred = p['sparse_pred']
                else:
                    sparse_pred += p['sparse_pred']

            sparse_pred /= len(models)
    #
    _, h, w = data_dict[0]['arr'].shape
    z_prob = sparse_pred[0].sigmoid().cpu()

    # print(xy_pred.shape)
    # n_seg, 2, 5, 3
    n_seg = len(index_info)
    pred_keypoints = np.zeros((n_seg, 2, 5, 2))
    for n in range(n_seg):
        z0, z1 = index_info[n]
        z_len = z1 - z0
        zs = coords[z0:z1]
        prob = z_prob[z0:z1]

        for i in range(5):
            left_conf = prob[:, 0, i].max()
            right_conf = prob[:, 1, i].max()
            recover_left_z = (prob[:, 0, i] * zs).sum() / prob[:, 0, i].sum()
            recover_right_z = (prob[:, 1, i] * zs).sum() / prob[:, 1, i].sum()

            if recover_left_z < 0:
                recover_left_z = 0
            if recover_left_z > z_len - 1:
                recover_left_z = z_len - 1

            if recover_right_z < 0:
                recover_right_z = 0
            if recover_right_z > z_len - 1:
                recover_right_z = z_len - 1

            pred_keypoints[n, 0, i, 0] = recover_left_z / z_len
            pred_keypoints[n, 0, i, 1] = left_conf
            pred_keypoints[n, 1, i, 0] = recover_right_z / z_len
            pred_keypoints[n, 1, i, 1] = right_conf

    series_id_list = data_dict[0]['series_id_list']
    return study_id, series_id_list, pred_keypoints


# infer together with v24 for dataloader speed
def axial_v2_infer_xyz(models, data_dict):
    arr_origin_list = data_dict[0]['arr_origin_list']
    series_id_list = data_dict[0]['series_id_list']
    nseg = len(arr_origin_list)

    pred_dict = {}
    for n in range(nseg):
        arr = arr_origin_list[n]
        z_len, h, w = arr.shape
        arr = arr.unsqueeze(0).unsqueeze(0)
        arr = F.interpolate(arr, size=(96, 256, 256))
        bs = 1
        keypoints = None
        with torch.no_grad():
            with autocast:
                for i in range(len(models)):
                    p = models[i](arr).cpu().numpy().reshape(bs, 10, 3)
                    if keypoints is None:
                        keypoints = p
                    else:
                        keypoints += p
        keypoints = keypoints / len(models)
        series_id = int(series_id_list[n])
        pred_dict[series_id] = {
            'points': keypoints[0],
            'd': z_len,
        }
    return pred_dict


def axial_v24_infer_xy(xy_models, pred_keypoints, data_dict):
    arr_origin_list = data_dict[0]['arr_origin_list']
    nseg, _, _, _ = pred_keypoints.shape  # n_seg, 2, 5, 2
    imgs = []
    for n in range(nseg):
        arr = arr_origin_list[n]
        z_len, h, w = arr.shape
        for idx in range(2):
            for level in range(5):
                z = pred_keypoints[n, idx, level, 0]
                z = int(np.round(z * z_len))
                if z < 0:
                    z = 0
                if z > z_len - 1:
                    z = z_len - 1
                imgs.append(arr[z].unsqueeze(0))
    # imgs: n_seg * 2 * 5
    imgs = torch.cat(imgs, dim=0).unsqueeze(1)
    # print('imgs: ', imgs.shape)
    xy_pred = None
    for i in range(len(xy_models)):
        with torch.no_grad():
            with autocast:
                p = xy_models[i](imgs)
                if xy_pred is None:
                    xy_pred = p
                else:
                    xy_pred += p
    xy_pred = xy_pred / len(xy_models)
    xy_pred = xy_pred.reshape(nseg, 2, 5, 4).cpu().numpy()
    # n_seg, 2, 5, 2
    xy_pred_final = np.concatenate(
        (xy_pred[:, 0:1, :, :2],  # left
         xy_pred[:, 1:2, :, 2:]),  # right
        axis=1)
    xy_pred_final = xy_pred_final / 512
    # n_seg, 2, 5, 4
    pred_keypoints = np.concatenate((xy_pred_final, pred_keypoints), axis=-1)
    return pred_keypoints


def load_axial_v24_level_models(
        model_dir='/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/v24/pretrain_axial_level_cls/convnext_small.in12k_ft_in1k_384/'
):
    fns = [
        model_dir + '/best_fold_0_ema.pt',
        model_dir + '/best_fold_1_ema.pt',
        model_dir + '/best_fold_2_ema.pt',
        model_dir + '/best_fold_3_ema.pt',
        model_dir + '/best_fold_4_ema.pt',
    ]
    models = []
    for fn in fns:
        print('load: ', fn)
        model = Axial_Level_Cls_Model_for_Test('convnext_small.in12k_ft_in1k_384',
                                               pretrained=False).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models.append(model)
    return models


def load_axial_v24_xy_models(
        model_dir='/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/v24/axial_2d_keypoints/densenet161_lr_0.0006/'
):
    fns = [
        model_dir + '/best_fold_0_ema.pt',
        model_dir + '/best_fold_1_ema.pt',
        model_dir + '/best_fold_2_ema.pt',
        model_dir + '/best_fold_3_ema.pt',
        model_dir + '/best_fold_4_ema.pt',
    ]
    models = []
    for fn in fns:
        print('load: ', fn)
        model = RSNA24Model_Keypoint_2D('densenet161',
                                        pretrained=False,
                                        num_classes=4).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models.append(model)
    return models


def load_3d_keypoints_models(model_dir, num_classes=15, is_parallel=False, ):
    models = []
    for fold in range(5):
        model = RSNA24Model_Keypoint_3D('densenet161',
                                        in_chans=1,
                                        pretrained=True,
                                        num_classes=num_classes).cuda()

        model.load_state_dict(
            torch.load(f'./{model_dir}/best_fold_{fold}_ema.pt'))

        model.eval()
        if is_parallel:
            model = nn.DataParallel(model)
        models.append(model)
    return models
