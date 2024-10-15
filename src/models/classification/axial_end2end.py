# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F
from src.utils.dicom import convert_to_8bit_lhw_version
from src.utils.comm import create_dir
import json
import pydicom
import glob
from skimage.transform import resize
from src.utils.dicom import convert_to_8bit_lhw_version


# window_center , window_width, intercept, slope = get_windowing(data)
# data.pixel_array = data.pixel_array * slope + intercept
# min_value = window_center - window_width // 2
# max_value = window_center + window_width // 2
# data.pixel_array.clip(min_value, max_value, out=data.pixel_array)

def load_dicom_axial(dicom_folder,
                     plane='axial',
                     reverse_sort=True,
                     img_size=512):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
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

    PixelSpacing = np.asarray([d.PixelSpacing for d in dicoms]).astype("float")[idx]
    SliceThickness = np.asarray([d.SliceThickness for d in dicoms]).astype("float")[idx]
    SpacingBetweenSlices = np.asarray([d.SpacingBetweenSlices for d in dicoms]).astype("float")[idx]
    SliceLocation = np.asarray([d.SliceLocation for d in dicoms]).astype("float")[idx]
    PatientPosition = dicoms[0].PatientPosition

    dicom_instance_numbers_to_idx = {}
    for i, ins in enumerate(dicom_instance_numbers):
        dicom_instance_numbers_to_idx[ins] = i
    meta = {
        "ImagePositionPatient": ipp,
        'SliceLocation': SliceLocation,
        "PixelSpacing": PixelSpacing,
        "SliceThickness": SliceThickness,
        "SpacingBetweenSlices": SpacingBetweenSlices,
        "PatientPosition": PatientPosition,

        "instance_num_to_shape": instance_num_to_shape,
        "dicom_instance_numbers": dicom_instance_numbers,
        "dicom_instance_numbers_to_idx": dicom_instance_numbers_to_idx,
    }
    return array, meta


def rescale_keypoints_by_meta(keypoints, meta, img_size=512):
    dicom_instance_numbers_to_idx = meta['dicom_instance_numbers_to_idx']
    instance_num_to_shape = meta['instance_num_to_shape']
    rescaled_keypoints = -np.ones_like(keypoints)
    for i in range(len(keypoints)):
        x, y, ins_num = keypoints[i]
        # no labeled
        if x < 0:
            continue
        origin_w, origin_h = instance_num_to_shape[ins_num]
        x = img_size / origin_w * x
        y = img_size / origin_h * y
        z = dicom_instance_numbers_to_idx[ins_num]
        rescaled_keypoints[i, 0] = x
        rescaled_keypoints[i, 1] = y
        rescaled_keypoints[i, 2] = z
    return rescaled_keypoints


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


def _gen_label_by_keypoints(pts, mean_margin, z_len, label=None):
    # pts (5, 3)
    if label is None:
        label = -100 * np.ones(z_len)

    margin = mean_margin / 4
    for level, p in enumerate(pts):
        z = p[2]
        if z < 0:
            assert z ==-1
            continue
        start_idx = int(np.round(z - margin))
        if start_idx < 0:
            start_idx = 0
        end_idx = int(np.round(z + margin))
        if end_idx > z_len - 1:
            end_idx = z_len - 1
        label[start_idx:end_idx] = level
    return label


class Axial_Level_Dataset(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None):
        super(Axial_Level_Dataset, self).__init__()
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
        for _, row in df.iterrows():
            study_id = row['study_id']
            label = row[1:].values.astype(np.int64)
            label = label[15:].reshape(2, 5)
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()

            for series_id in series_id_list:
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
                if len(coord_sub_df) > 0:
                    self.samples.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'label': label
                    })

        print('samples: ', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_axial_coord(self, study_id, series_id):
        coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
        coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
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

    def __getitem__(self, idx):
        item = self.samples[idx]
        study_id = item['study_id']
        series_id = item['series_id']
        cls_label = item['label']

        img_size = 160
        keypoints = self.get_axial_coord(study_id, series_id)
        # print(keypoints)
        dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
        arr, meta = load_dicom_axial(dicom_dir, img_size=img_size)
        keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=img_size)
        # print('rescaled: ', keypoints)
        # print('PatientPosition: ', meta['PatientPosition'])
        # print('arr shape: ', arr.shape)
        # resample by SpacingBetweenSlices
        SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]
        target_spacing = 2.0
        z_scale = SpacingBetweenSlices / target_spacing
        depth = int(arr.shape[0] * z_scale)
        # target_size = (depth, img_size, img_size)
        target_size = (depth, img_size, img_size)
        arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                            size=target_size).numpy().squeeze()

        #keypoints[:, 2] = keypoints[:, 2] * z_scale
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] != -1:
                keypoints[i, 2] = keypoints[i, 2] * z_scale

        keypoints = keypoints.reshape(5, 2, 3)
        left_keypoints = keypoints[:, 0]
        right_keypoints = keypoints[:, 1]
        mean_margin_left = _get_keypoint_mean_margin(left_keypoints)
        mean_margin_right = _get_keypoint_mean_margin(right_keypoints)


        z_len = arr.shape[0]
        if mean_margin_left != -1 and mean_margin_right != -1:
            mean_margin = (mean_margin_right + mean_margin_left) / 2
        elif mean_margin_left == -1 and mean_margin_right != -1:
            mean_margin = mean_margin_right
        elif mean_margin_right == -1 and mean_margin_left != -1:
            mean_margin = mean_margin_left
        else:
            mean_margin = 9

        label = _gen_label_by_keypoints(left_keypoints, mean_margin, z_len)
        label = _gen_label_by_keypoints(right_keypoints, mean_margin, z_len, label)
        if self.transform is not None:
            arr = arr.transpose(1, 2, 0)
            arr = self.transform(image=arr)['image']
            arr = arr.transpose(2, 0, 1)
        #print(study_id, label)
        for i in range(5):
            if left_keypoints[i,  2] < 0:
                cls_label[0, i] = -100
            if right_keypoints[i, 2] < 0:
                cls_label[1, i] = -100
        cls_label = cls_label.reshape(-1)
        return {
            'img': arr.astype(np.float32),
            'label': label.astype(np.int64),
            'PatientPosition': meta['PatientPosition'],
            'cls_label': cls_label
        }


if __name__ == '__main__':
    import tqdm

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    #df = df[df['study_id']==26342422]
    dset = Axial_Level_Dataset(data_root, df)
    dloader = DataLoader(dset, num_workers=12, batch_size=1)
    lens = []
    PatientPosition_list = []
    for d in tqdm.tqdm(dloader):
        arr = d['img']
        lens.append(arr[0].shape[0])
        PatientPosition = d['PatientPosition'][0]
        PatientPosition_list.append(PatientPosition)

    print(pd.DataFrame(lens).describe())
    print(pd.DataFrame(PatientPosition_list).describe())
    print(pd.DataFrame(PatientPosition_list).value_counts())
