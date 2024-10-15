# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import tqdm
import cv2
import numpy as np
import pickle
import pandas as pd
from sklearn.cluster import KMeans

from src.utils.dicom import load_dicom_stack_lhw_version
from src.utils.comm import create_dir
import json
from train_scripts.data_path import DATA_ROOT
data_dir = DATA_ROOT


df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
image_dir = f"{data_dir}/train_images/"
save_dir = f"{data_dir}/train_images_preprocessed/"
create_dir(save_dir)

def unique_blocks(arr):
    n = []
    for i in arr:
        if i not in n:
            n.append(i)
    return n

from concurrent.futures import ProcessPoolExecutor


def process_dicom(row):
    row = row[1]
    plane = "sagittal"
    reverse_sort = False
    if row['series_description'] == "Axial T2":
        plane = "axial"
        reverse_sort = True

    dicom_dir = os.path.join(image_dir, str(row['study_id']), str(row['series_id']))
    dicom_volumn = load_dicom_stack_lhw_version(dicom_dir, plane, reverse_sort,
                                                img_size=512)

    sub_save_dir = os.path.join(save_dir, str(row['study_id']), str(row['series_id']))
    create_dir(sub_save_dir)

    array = dicom_volumn['array']
    for j in range(array.shape[0]):
        dst_path = f'{sub_save_dir }/{j:03d}.png'
        img = array[j]
        cv2.imwrite(dst_path, img)

    meta = {
        'instance_num_to_shape': dicom_volumn['instance_num_to_shape'],
        'dicom_instance_numbers': dicom_volumn['dicom_instance_numbers']
    }
    json.dump(meta, open(f'{sub_save_dir }/meta_info.json', 'w'))

    meta2 = {
        'instance_num_to_shape': dicom_volumn['instance_num_to_shape'],
        'dicom_instance_numbers': dicom_volumn['dicom_instance_numbers'],
        "positions": dicom_volumn['positions'],
        "PixelSpacing":  dicom_volumn['PixelSpacing'],
        "SliceThickness": dicom_volumn['SliceThickness'],
        "SpacingBetweenSlices": dicom_volumn['SpacingBetweenSlices'],
        'SliceLocation': dicom_volumn['SliceLocation'],
        'PatientPosition': dicom_volumn['PatientPosition']
    }
    # if row['series_description'] == "Axial T2":
    #     positions = dicom_volumn['positions']
    #     kmeans_model = KMeans(n_clusters=5, random_state=2024).fit(positions)
    #     label_ = kmeans_model.labels_
    #     result = unique_blocks(label_)
    #     level_to_index = {}
    #     for level, r in enumerate(result):
    #         idx = np.where(label_ == r)
    #         level_to_index[level] = idx
    #     meta2['level_to_index'] = level_to_index

    pickle.dump(meta2, open(f'{sub_save_dir }/meta_info_2.pkl', 'wb'))


with ProcessPoolExecutor(max_workers=12) as executor:
    list(tqdm.tqdm(executor.map(process_dicom, df.iterrows(), chunksize=1),
                   total=df.shape[0]))
