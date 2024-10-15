import os
import pickle

import timm_3d

#os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import cv2

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

def select_elements(K, N, randomize=False):
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

####
# lhwcv dataset related

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



def build_axial_group_idxs(pred_keypoints_infos0, img_size=512):
    dict_axial_group_idxs = {}

    # pred_keypoints_infos0 = pickle.load(
    #     open(f'{data_root}/v2_axial_3d_keypoints_model0.pkl', 'rb'))

    for study_id in pred_keypoints_infos0.keys():
        for sid in pred_keypoints_infos0[study_id].keys():
            pred_info0 = pred_keypoints_infos0[study_id][sid]
            predict_keypoints = pred_info0['points'].reshape(-1, 3)
            depth = int(pred_info0['d'])

            scale = img_size / 4.0
            scale_z = depth / 16.0
            predict_keypoints[:, :2] = predict_keypoints[:, :2] * scale
            predict_keypoints[:, 2] = predict_keypoints[:, 2] * scale_z

            pred_z = predict_keypoints[:, 2].reshape(2, 5)
            pred_z = np.round(pred_z.mean(axis=0))
            group_idxs = gen_level_group_idxs(pred_z, depth)
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

class RSNA24DatasetTest_LHW_keypoint_3D_Axial(Dataset):
    def __init__(self,
                 data_root,
                 test_series_descriptions_fn,
                 study_ids,
                 transform=None,
                 image_dir=None):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root, test_series_descriptions_fn)

        self.samples = []
        for study_id in study_ids:
            info = self.aux_info[study_id]
            sids = info['des_to_sid']['Axial T2']

            for sid in sids:
                self.samples.append({
                    'study_id': study_id,
                    'sid': sid,
                })
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform
        self.img_size = 256
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

    def __len__(self):
        return len(self.samples)

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
        if volume is None:
            volume = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
        return volume, sid

    def __getitem__(self, idx):
        a = self.samples[idx]
        study_id = a['study_id']
        sid = a['sid']

        # Axial T2
        a_t2, sid = self.get_3d_volume(study_id, sid, 'Axial T2')

        if self.transform is not None:
            # to h,w,c
            a_t2 = a_t2.transpose(1, 2, 0)
            a_t2 = self.transform(image=a_t2)['image']
            a_t2 = a_t2.transpose(2, 0, 1)

        x = a_t2
        x = x.astype(np.float32)
        depth = x.shape[0]
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(96, 256, 256)).squeeze()
        return x, study_id, sid, depth




class RSNA24DatasetTest_LHW_V2(Dataset):
    def __init__(self,
                 data_root,
                 study_ids,
                 test_series_descriptions_fn,
                 study_id_to_pred_keypoints_sag=None,
                 study_id_to_pred_keypoints_axial=None,
                 pred_sag_keypoints_infos_3d_t1=None,
                 with_axial=True,
                 transform=None,
                 axial_transform=None,
                 gt_df_maybe=None,
                 image_dir=None,
                 cache_dir=None,
                 ):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root,test_series_descriptions_fn)
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        self.transform = transform
        if axial_transform is None:
            axial_transform = A.Compose([
                #A.Resize(512, 512),
                A.CenterCrop(256, 256),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.axial_transform = axial_transform
        self.img_size = 512
        self.gt_df_maybe = gt_df_maybe
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

        self.with_axial = with_axial
        if with_axial:
            self.dict_axial_group_idxs = build_axial_group_idxs(study_id_to_pred_keypoints_axial)

        self.study_id_to_pred_keypoints_sag = study_id_to_pred_keypoints_sag
        self.crop_sag = False
        if study_id_to_pred_keypoints_sag is not None:
            self.crop_sag = True

        self.sag_total_slices = 10
        # for example s_t2 keep only the center 5 imgs
        # Spinal Canal Stenosis
        self.sag_t2_index_range = [3, 8]
        # Left Neural Foraminal Narrowing
        self.sag_t1_left_index_range = [5, 10]
        # Right Neural Foraminal Narrowing
        self.sag_t1_right_index_range = [0, 5]
        self.cache_dir = cache_dir
        self.pred_sag_keypoints_infos_3d_t1 = pred_sag_keypoints_infos_3d_t1

    def __len__(self):
        return len(self.study_ids)

    def slice_volume(self, volume, index, bbox=None):
        arr = []
        for i in index:
            a = volume[i]
            if bbox is not None:
                a = a[bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
                a = cv2.resize(a, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            arr.append(a)
        arr = np.array(arr, np.uint8)
        return arr

    def gen_slice_index_Sagittal(self, length):
        # step = length / 15.0
        # st = length / 2.0 - 6.5 * step
        # end = length + 0.0001
        step = length / 10.0
        st = length / 2.0 - 4.0 * step
        end = length + 0.0001
        slice_indexes = []
        for i in np.arange(st, end, step):
            inx = max(0, int((i - 0.5001).round()))
            slice_indexes.append(inx)
        return slice_indexes

    def gen_slice_index_Axial(self, length):
        # step = length / 15.0
        # st = length / 2.0 - 6.5 * step
        # end = length + 0.0001
        step = length / 10.0
        st = length / 2.0 - 4.0 * step
        end = length + 0.0001
        slice_indexes = []
        for i in np.arange(st, end, step):
            inx = max(0, int((i - 0.5001).round()))
            slice_indexes.append(inx)
        return slice_indexes

    # def load_axial(self, study_id, sid):
    #     return load_dicom_line_par(self.image_dir, study_id, sid)

    def get_3d_volume(self, study_id, sids, series_description):
        # TODO use multi when test
        sid = sids[0]

        prefix = 'sag_t2'
        if series_description == 'Sagittal T1':
            prefix = 'sag_t1'
        fn1 = f'{self.cache_dir}/{prefix}_{study_id}_{sid}_img.npz'
        if self.cache_dir is not None and series_description != "Axial T2":
            if os.path.exists(fn1):
                arr = np.load(fn1)['arr_0']
                return arr, sid
        if series_description!='Axial T2':
            print('[warn] load without cache!')
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))
        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            plane = "axial"
            reverse_sort = True
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']

        # if self.cache_dir is not None and series_description != "Axial T2":
        #     np.savez_compressed(fn1, volume)

        return volume, sid

    def crop_out_by_keypoints(self, volume, keypoints, img_size=128, G_IMG_SIZE=512):
        sub_volume_list = []
        att_mask_list = []
        for p in keypoints:
            x, y = int(p[0]), int(p[1])
            bbox = [x - img_size // 2,
                    y - img_size // 2,
                    x + img_size // 2,
                    y + img_size // 2]
            # bbox = np.clip(bbox, 0, 512)
            # bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

            # 如果bbox在边界，偏移它以使其具有正确的img_size大小
            if bbox[0] < 0:
                bbox[2] = img_size
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[3] = img_size
                bbox[1] = 0
            if bbox[2] > G_IMG_SIZE:
                bbox[0] = G_IMG_SIZE - img_size
                bbox[2] = G_IMG_SIZE
            if bbox[3] > G_IMG_SIZE:
                bbox[1] = G_IMG_SIZE - img_size
                bbox[3] = G_IMG_SIZE

            bbox = [int(e) for e in bbox]
            v = volume[:, bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()

            sub_volume_list.append(v)

        volume_crop = np.array(sub_volume_list)
        return volume_crop

    def crop_out_by_z(self,volume,
                           z,
                           z_imgs=3,
                           ):
        z0 = z - z_imgs // 2
        z1 = z + z_imgs // 2 + 1
        if z0 < 0:
            z0 = 0
            z1 = z_imgs
        if z1 > volume.shape[0]:
            z0 = volume.shape[0] - z_imgs
            z1 = volume.shape[0]

        v = volume[z0: z1].copy()
        return v

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        info = self.aux_info[study_id]
        des_to_sid = info['des_to_sid']
        # Sagittal T1
        if 'Sagittal T1' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T1 images'.format(study_id))
            s_t1 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
        else:

            volume, sid = self.get_3d_volume(study_id,
                                             des_to_sid['Sagittal T1'],
                                             'Sagittal T1')

            if volume is None:
                s_t1 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
            else:
                if self.pred_sag_keypoints_infos_3d_t1 is None:
                    index = self.gen_slice_index_Sagittal(length=len(volume))
                    s_t1 = self.slice_volume(volume, index, bbox=None)
                else:
                    if len(volume) < 10:
                        index = self.gen_slice_index_Sagittal(length=len(volume))
                        s_t1 = self.slice_volume(volume, index, bbox=None)
                    else:
                        s_t1 = volume

                    t1_pred_keypoints = self.pred_sag_keypoints_infos_3d_t1[study_id][sid].reshape(10, 3).astype(np.float32)

                    t1_pred_keypoints[:, 2] = s_t1.shape[0] * t1_pred_keypoints[:, 2]
                    t1_pred_keypoints = t1_pred_keypoints.reshape(2, 5, 3)
                    z_left = int(np.round(t1_pred_keypoints[0][:, 2].mean(axis=0)))
                    z_right = int(np.round(t1_pred_keypoints[1][:, 2].mean(axis=0)))

                    s_t1_left = self.crop_out_by_z(s_t1, z_left, z_imgs=5)
                    s_t1_right = self.crop_out_by_z(s_t1, z_right, z_imgs=5)
                    s_t1 = np.concatenate([s_t1_right, s_t1_left], axis=0)

        # Sagittal T2/STIR
        if 'Sagittal T2/STIR' not in des_to_sid.keys():
            print('[WARN] {} has no Sagittal T2/STIR images'.format(study_id))
            s_t2 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
        else:

            volume, sid = self.get_3d_volume(study_id,
                                             des_to_sid['Sagittal T2/STIR'],
                                             'Sagittal T2/STIR')

            if volume is None:
                s_t2 = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
            else:
                index = self.gen_slice_index_Sagittal(length=len(volume))
                s_t2 = self.slice_volume(volume, index, bbox=None)

        ret_dict = {
            'study_id': str(study_id)
        }
        if self.with_axial:
            levels = range(5)

            axial_sids = des_to_sid['Axial T2']
            axial_volumes = []
            for sid in axial_sids:
                volume, sid = self.get_3d_volume(study_id,
                                                 [sid],
                                                 'Axial T2')
                if volume is None:
                    volume = np.zeros((10, self.img_size, self.img_size), dtype=np.uint8)
                axial_volumes.append(volume)

            axial_imgs = []
            for level in levels:
                tmp_imgs = []
                for volume_, sid_ in zip(axial_volumes, axial_sids):
                    r = self.dict_axial_group_idxs[study_id][sid_]['group_idxs'][level]
                    indexes = list(range(r[0], r[1] + 1))
                    imgs = self.slice_volume(volume_, indexes, bbox=None)
                    imgs = imgs.transpose(1, 2, 0)

                    ind = select_elements(imgs.shape[-1], 6)
                    imgs_ = imgs[:, :, ind]
                    ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
                    ind_a = np.where(np.array(ind) + 1 >= imgs.shape[-1] - 1, imgs.shape[-1] - 1, np.array(ind) + 1)
                    imgs = np.stack([imgs[:, :, ind_b], imgs_, imgs[:, :, ind_a]], axis=2)

                    imgs = np.stack(
                        [self.axial_transform(image=imgs[:, :, :, i])['image'] for i in range(imgs.shape[-1])])
                    # 6, 3, h, w
                    imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2))).float()
                    tmp_imgs.append(imgs)
                    # break

                tmp_imgs = np.concatenate(tmp_imgs, axis=0)  # 6*num_series,3,h,w
                axial_imgs.append(tmp_imgs)

            # 5, 6, 3, h, w
            axial_imgs = np.array(axial_imgs)
            axial_imgs = axial_imgs.astype(np.float32)
            ret_dict['axial_imgs'] = torch.from_numpy(axial_imgs)


        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)
            # a_t2 = a_t2.transpose(1, 2, 0)

            s_t1 = self.transform(image=s_t1)['image']
            s_t2 = self.transform(image=s_t2)['image']
            # a_t2 = self.transform(image=a_t2)['image']

            s_t1 = s_t1.transpose(2, 0, 1)
            s_t2 = s_t2.transpose(2, 0, 1)
            # a_t2 = a_t2.transpose(2, 0, 1)

        if self.crop_sag:
            keypoints = self.study_id_to_pred_keypoints_sag[study_id]
            s_t1_keypoints = keypoints[0]
            s_t2_keypoints = keypoints[1]

            r = self.sag_t2_index_range
            s_t2 = s_t2[r[0]: r[1]]
            s_t2 = self.crop_out_by_keypoints(s_t2, s_t2_keypoints)

            r = self.sag_t1_left_index_range
            s_t1_left = s_t1[r[0]:r[1], :, :]
            r = self.sag_t1_right_index_range
            s_t1_right = s_t1[r[0]:r[1], :, :]

            s_t1_left = self.crop_out_by_keypoints(s_t1_left, s_t1_keypoints)
            s_t1_right = self.crop_out_by_keypoints(s_t1_right, s_t1_keypoints)

            x = np.concatenate([s_t2, s_t1_left, s_t1_right, s_t2, s_t2], axis=0)
            x = x.astype(np.float32)
            cond = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
            cond = np.array(cond, dtype=np.int64)
            ret_dict['img'] = torch.from_numpy(x)
            ret_dict['cond'] = cond
        else:
            x = np.concatenate([s_t1, s_t2], axis=0)
            x = x.astype(np.float32)
            ret_dict['img'] = torch.from_numpy(x)

        return ret_dict


def v2_collate_fn(batch):
    img = []
    axial_imgs = []
    study_id = []
    cond = []
    max_n = 0
    n_list = []
    for item in batch:
        cond.append(torch.from_numpy(item['cond']).long())
        img.append(item['img'])
        a = item['axial_imgs']
        _, n, _, h, w = a.shape
        n_list.append(n)
        if n > max_n:
            max_n = n
        axial_imgs.append(a)
        study_id.append(item['study_id'])

    if max_n !=6:
        bs = len(axial_imgs)
        padded_axial_imgs = torch.zeros((bs, 5, max_n, 3, 256, 256), dtype=torch.float32)
        for i in range(bs):
            padded_axial_imgs[i, :, :n_list[i]] = axial_imgs[i]
        axial_imgs = padded_axial_imgs
    else:
        axial_imgs = torch.stack(axial_imgs, dim=0)

    return {'img': torch.stack(img, dim=0),
            'cond': torch.stack(cond, dim=0),
            'axial_imgs': axial_imgs,
            'study_id': study_id,
            'n_list': torch.LongTensor(n_list),
            }

G_IMG_SIZE = 512
class RSNA24DatasetTest_LHW_keypoint_3D_Saggital(Dataset):
    def __init__(self,
                 data_root,
                 test_series_descriptions_fn,
                 study_ids,
                 transform=None,
                 image_dir=None,
                 cache_dir=None,
                 ):
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
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.study_ids)

    def get_3d_volume(self, study_id, sid, series_description):
        # TODO use multi when test
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))

        prefix = 'sag_t2'
        if series_description == 'Sagittal T1':
            prefix = 'sag_t1'
        fn1 = f'{self.cache_dir}/{prefix}_{study_id}_{sid}_img.npz'
        if self.cache_dir is not None and series_description != "Axial T2":
            if os.path.exists(fn1):
                arr = np.load(fn1)['arr_0']
                return arr, sid


        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            plane = "axial"
            reverse_sort = True
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']

        if volume is not None:
            if self.cache_dir is not None and series_description != "Axial T2":
                np.savez_compressed(fn1, volume)

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
            s_t1,_ = self.get_3d_volume(study_id, sid, "Sagittal T1")
            if s_t1 is None:
                s_t1 = np.zeros((10, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)

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
            if s_t2 is None:
                s_t2 = np.zeros((10, G_IMG_SIZE, G_IMG_SIZE), dtype=np.uint8)

        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t1 = self.transform(image=s_t1)['image']
            s_t1 = s_t1.transpose(2, 0, 1)

            s_t2 = s_t2.transpose(1, 2, 0)
            s_t2 = self.transform(image=s_t2)['image']
            s_t2 = s_t2.transpose(2, 0, 1)

        s_t1 = s_t1.astype(np.float32)
        s_t1 = torch.from_numpy(s_t1).unsqueeze(0).unsqueeze(0)
        s_t1 = F.interpolate(s_t1, size=(48, 256, 256)).squeeze(0)

        s_t2 = s_t2.astype(np.float32)
        s_t2 = torch.from_numpy(s_t2).unsqueeze(0).unsqueeze(0)
        s_t2 = F.interpolate(s_t2, size=(48, 256, 256)).squeeze(0)

        x = torch.cat((s_t1, s_t2), dim=0)

        return x, study_id