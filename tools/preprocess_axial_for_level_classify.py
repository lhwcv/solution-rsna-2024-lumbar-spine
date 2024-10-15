# -*- coding: utf-8 -*-
import os

os.environ['OMP_NUM_THREADS'] = '1'
import torch
import tqdm
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
from concurrent.futures import ProcessPoolExecutor

data_dir = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
image_dir = f"{data_dir}/train_images/"
save_dir = f"{data_dir}/train_images_preprocessed_axial/imgs/"
create_dir(save_dir)
df = df[df['series_description'].isin(['Axial T2'])]
coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
level2int = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}
coord_label_df = coord_label_df.replace(level2int)


def get_axial_coord(study_id, series_id):
    coord_sub_df = coord_label_df[coord_label_df['study_id'] == study_id]
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


def _is_keypoint_contain_5_level(pts):
    # 5, 3
    z0 = pts[0, 2]
    z1 = pts[4, 2]
    if z0 > 0 and z1 > 0:
        return True
    return False


def _find_keypoint_contain_level_range(pts):
    # 5, 3
    start_idx = -1
    for i in range(len(pts)):
        if pts[i, 2] > 0:
            start_idx = i
            break
    end_idx = -1
    for i in range(len(pts) - 1, 0):
        if pts[i, 2] > 0:
            end_idx = i
            break
    return start_idx, end_idx


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


# def gen_3_slice_index(start_idx, end_idx):
#     if end_idx - start_idx == 0:
#         return [[start_idx, start_idx, start_idx]]
#     indexes = []
#
#     def _clamp(x):
#         if x < start_idx:
#             x = start_idx
#         if x > end_idx:
#             x = end_idx
#         return x
#
#     for i in range(start_idx, end_idx + 1):
#         i0 = _clamp(i - 1)
#         i1 = _clamp(i)
#         i2 = _clamp(i + 1)
#         indexes.append([i0, i1, i2])
#     return indexes


def gen_5_slice_index(start_idx, end_idx):
    if end_idx - start_idx == 0:
        return [[start_idx, start_idx, start_idx, start_idx, start_idx]]
    indexes = []

    def _clamp(x):
        if x < start_idx:
            x = start_idx
        if x > end_idx:
            x = end_idx
        return x

    for i in range(start_idx, end_idx + 1, 2):
        i0 = _clamp(i - 2)
        i1 = _clamp(i - 1)
        i2 = _clamp(i)
        i3 = _clamp(i + 1)
        i4 = _clamp(i + 2)
        indexes.append([i0, i1, i2, i3, i4])
    return indexes

def _crop_by_keypoints_z(arr, pts, mean_margin):
    margin = mean_margin / 4
    imgs_dict = {}
    for level, p in enumerate(pts):
        z = p[2]
        if z == -1:
            continue
        start_idx = int(np.round(z - margin))
        if start_idx <0:
            start_idx=0
        end_idx = int(np.round(z + margin))
        if end_idx > arr.shape[0]-1:
            end_idx = arr.shape[0]-1

        #print(f'level: {level} ', start_idx, end_idx)
        indexes = gen_5_slice_index(start_idx, end_idx)
        imgs = []
        for ind in indexes:
            imgs.append(arr[ind].copy().transpose(1,2, 0))
        imgs_dict[f'level_{level}'] = imgs
    return imgs_dict

def process_group(g: pd.DataFrame, img_size=512):
    assert g['study_id'].nunique() == 1, print(g['study_id'].unique())
    study_id = g['study_id'].iloc[0]
    series_id_list = g['series_id'].to_list()

    for series_id in series_id_list:
        keypoints = get_axial_coord(study_id, series_id)
        # print(keypoints)
        dicom_dir = f'{image_dir}/{study_id}/{series_id}/'
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
        #target_size = (depth, img_size, img_size)
        target_size = (depth, 256, 256)
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
        if mean_margin_left != -1 and mean_margin_right != -1:
            mean_margin = (mean_margin_right + mean_margin_left) / 2
        elif mean_margin_left == -1 and mean_margin_right != -1:
            mean_margin = mean_margin_right
        elif mean_margin_right == -1 and mean_margin_left != -1:
            mean_margin = mean_margin_left
        else:
            mean_margin = 9
        imgs_dict_left = _crop_by_keypoints_z(arr, left_keypoints, mean_margin)
        imgs_dict_right = _crop_by_keypoints_z(arr, right_keypoints, mean_margin)
        for k, imgs in imgs_dict_left.items():
            for n, img in enumerate(imgs):
                fn = f'{save_dir}/{study_id}_{series_id}_left_{k}_{n}.npz'
                np.savez_compressed(fn, img)
                #cv2.imwrite(fn, img)
        for k, imgs in imgs_dict_right.items():
            for n, img in enumerate(imgs):
                fn = f'{save_dir}/{study_id}_{series_id}_right_{k}_{n}.npz'
                np.savez_compressed(fn, img)
                #cv2.imwrite(fn, img)

groups = df.groupby(['study_id'])
sub_dfs = []
for i, g in tqdm.tqdm(groups):
    sub_dfs.append(g.copy())

# 创建一个ProcessPoolExecutor实例
with ProcessPoolExecutor() as executor:
    # 使用tqdm显示进度
    list(tqdm.tqdm(executor.map(process_group, sub_dfs), total=len(sub_dfs)))