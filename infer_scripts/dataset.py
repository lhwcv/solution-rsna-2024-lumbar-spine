# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import cv2

import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pydicom
from skimage.transform import resize
from sklearn.cluster import KMeans
import torch.nn.functional as F

####
# patriot dataset related
def select_elements(K, N):
    lst = list(range(K))
    length = K

    if length <= N:
        # K+1 <= N の場合
        repeat_times = (N // length) + 1
        lst = sorted(lst * repeat_times)
    interval = len(lst) / N
    result = [lst[int(i * interval)] for i in range(N)]
    return result


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    pos = dicom.ImagePositionPatient
    return data, pos


def scale(images):
    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)
    return images


def unique_blocks(arr):
    # 差分を取り、0でない要素のインデックスを取得
    n = []
    for i in arr:
        if i not in n:
            n.append(i)
    return n


def load_dicom_line_par(img_dir, study_id, sid):
    path = f"/{img_dir}/{study_id}/{sid}"
    t_paths = sorted(glob.glob(os.path.join(path, "*")), key=lambda x: int(x.split('/')[-1].split(".")[0]))

    images = []
    pos_ = []
    for filename in t_paths:
        data, pos = load_dicom(filename)
        images.append(data)
        pos_.append(np.array(pos))
    pos_ = np.stack(pos_)
    SORT = 2
    sort_ = np.array([i[SORT] for i in pos_])
    pos_ = pos_[np.argsort(sort_)]
    try:
        images = np.array(images)
        images = images[np.argsort(sort_)]
    except:
        t_paths = np.array(t_paths)[np.argsort(sort_)]
        images = []
        for filename in t_paths:
            data, pos = load_dicom(filename)
            images.append(data)


    kmeans_model = KMeans(n_clusters=5, random_state=2024).fit(pos_)
    label_ = kmeans_model.labels_

    result = unique_blocks(label_)

    tmp_ = []
    for l, r in enumerate(result):
        idx = np.where(label_ == r)
        try:
            tmp = images[idx]
        except:
            tmp = [images[j] for j in idx[0]]
        tmp = np.stack(tmp, -1)

        tmp = scale(tmp)
        tmp_.append(tmp)

    return tmp_


####
# lhwcv dataset related

def get_test_des_to_sids(data_root, test_series_descriptions_fn='test_series_descriptions.csv'):
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


def load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort=False,
                                 img_size=512):
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
        arr = resize(arr, (img_size, img_size))
        array.append(arr)
    array = np.array(array)
    array = array[idx]
    array = convert_to_8bit_lhw_version(array)
    # array = resize_volume(array, 512, 512)

    return {"array": array, }


class RSNA24DatasetTest_LHW_V2(Dataset):
    def __init__(self,
                 data_root,
                 study_ids,
                 transform=None,
                 axial_transform=None,
                 gt_df_maybe=None,
                 image_dir=None,
                 img_size=512,
                 with_axial=True,
                 test_series_descriptions_fn='test_series_descriptions.csv'):
        self.study_ids = study_ids
        self.aux_info = get_test_des_to_sids(data_root, test_series_descriptions_fn)
        self.with_axial = with_axial
        if transform is None:
            transform = A.Compose([
                A.Normalize(mean=0.5, std=0.5)
            ])
        self.transform = transform
        if axial_transform is None:
            axial_transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CenterCrop(307, 307),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.axial_transform = axial_transform
        self.in_channels = 30
        self.img_size = img_size
        self.gt_df_maybe = gt_df_maybe
        if image_dir is None:
            self.image_dir = os.path.join(data_root, 'test_images')
        else:
            self.image_dir = image_dir

        #self.kmeans_model = KMeans(n_clusters=5, random_state=2024)

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

    def load_axial(self, study_id, sid):
        return load_dicom_line_par(self.image_dir, study_id, sid)

    def get_3d_volume(self, study_id, sids, series_description):
        # TODO use multi when test
        sid = sids[0]
        dicom_folder = os.path.join(self.image_dir, str(study_id), str(sid))
        plane = "sagittal"
        reverse_sort = False
        if series_description == "Axial T2":
            # plane = "axial"
            # reverse_sort = True
            return self.load_axial(study_id, sid), sid
        volume = load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort)
        volume = volume['array']

        return volume, sid

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
                index = self.gen_slice_index_Sagittal(length=len(volume))
                s_t1 = self.slice_volume(volume, index, bbox=None)

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

        #print('here')
        if self.with_axial:
            # Axial T2
            volume, sid = self.get_3d_volume(study_id,
                                             des_to_sid['Axial T2'],
                                             'Axial T2')
            # print('load Axial T2 done!')
            # sid = des_to_sid['Axial T2'][0]
            axial_imgs = []
            for level in range(5):
                # data_dir = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
                # imgs = np.load(f"{data_dir}/t2A_each/{study_id}__{sid}__{4 - level}.npy")
                imgs = volume[4 - level]
                ind = select_elements(imgs.shape[-1], 6)
                imgs_ = imgs[:, :, ind]
                ind_b = np.where(np.array(ind) - 1 < 0, 0, np.array(ind) - 1)
                ind_a = np.where(np.array(ind) + 1 >= imgs.shape[-1] - 1, imgs.shape[-1] - 1, np.array(ind) + 1)
                imgs = np.stack([imgs[:, :, ind_b], imgs_, imgs[:, :, ind_a]], axis=2)

                imgs = np.stack([self.axial_transform(image=imgs[:, :, :, i])['image'] for i in range(imgs.shape[-1])])
                # 6, 3, h, w
                imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2))).float()
                axial_imgs.append(imgs)
            # 5, 6, 3, h, w
            axial_imgs = np.array(axial_imgs)
            axial_imgs = axial_imgs.astype(np.float32)


        if self.transform is not None:
            # to h,w,c
            s_t1 = s_t1.transpose(1, 2, 0)
            s_t2 = s_t2.transpose(1, 2, 0)

            s_t1 = self.transform(image=s_t1)['image']
            s_t2 = self.transform(image=s_t2)['image']

            s_t1 = s_t1.transpose(2, 0, 1)
            s_t2 = s_t2.transpose(2, 0, 1)

        x = np.concatenate([s_t1, s_t2], axis=0)
        x = x.astype(np.float32)
        if self.with_axial:
            return x, axial_imgs, str(study_id)
        else:
            return x, str(study_id)


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
            # if study_id == 3429409220:
            #     print(sids)
            #     exit(0)
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