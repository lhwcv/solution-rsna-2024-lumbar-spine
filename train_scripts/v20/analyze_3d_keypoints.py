# -*- coding: utf-8 -*-
import os
import tqdm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
from train_scripts.v20.sag_data import Sag_3D_Point_Dataset
import albumentations as A
import torch.nn as nn


def xyz_pixel_err(pred, gt, mask):
    # bs, _ = pred.shape
    bs = 1
    pred = pred.reshape(bs, -1, 5, 3)
    gt = gt.reshape(bs, -1, 5, 3)
    mask = mask.reshape(bs, -1, 5, 3)

    pred = pred * mask
    gt = gt * mask

    # xy@512, z@32
    pred[:, :, :, :2] = 512 * pred[:, :, :, :2]
    gt[:, :, :, :2] = 512 * gt[:, :, :, :2]

    pred[:, :, :, 2] = 32 * pred[:, :, :, 2]
    gt[:, :, :, 2] = 32 * gt[:, :, :, 2]

    dis_func = nn.L1Loss(reduction='none')

    xy_err = dis_func(pred[:, :, :, :2],
                      gt[:, :, :, :2]).sum() / (mask[:, :, :, :2].sum())
    z_err = dis_func(pred[:, :, :, 2],
                     gt[:, :, :, 2]).sum() / (mask[:, :, :, 2].sum())
    return xy_err.item(), z_err.item()


if __name__ == '__main__':
    series_description = 'Sagittal T1'  # Sagittal T2/STIR
    #series_description = 'Sagittal T2/STIR'
    save_prefix = series_description.replace("/", "").replace(" ", "")
    err_df_name = f'{save_prefix}_err_df.csv'
    if False:#os.path.exists(err_df_name):
        df = pd.read_csv(err_df_name)
    else:
        data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
        df = pd.read_csv(f'{data_root}/train.csv')
        df = df.fillna(-100)
        label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        df = df.replace(label2id)

        if series_description == 'Sagittal T1':
            pred_fn = f'{data_root}/v20_sag_T1_3d_keypoints_oof_densenet161.pkl'
        else:
            pred_fn = f'{data_root}/v20_sag_T2_3d_keypoints_oof_densenet161.pkl'

        dset = Sag_3D_Point_Dataset(data_root, df, transform=None,
                                    phase='test', series_description=series_description)
        dloader = DataLoader(dset, num_workers=12)
        all_pred = pickle.load(open(pred_fn, 'rb'))

        data_frames = {
            'study_id': [],
            'series_id': [],
            'xy_err': [],
            'z_err': []
        }

        for d in tqdm.tqdm(dloader):
            gt = d['keypoints'][0].reshape(-1, 5, 3)
            mask = d['mask'][0].reshape(-1, 5, 3)
            study_id = int(d['study_id'][0])
            series_id = int(d['series_id'][0])
            origin_depth = int(d['origin_depth'][0])
            pred = torch.from_numpy(all_pred[study_id][series_id])
            xy_err, z_err = xyz_pixel_err(pred, gt, mask)
            z_err = z_err * origin_depth / 32.0
            data_frames['study_id'].append(study_id)
            data_frames['series_id'].append(series_id)
            data_frames['xy_err'].append(xy_err)
            data_frames['z_err'].append(z_err)

        df = pd.DataFrame(data_frames)
        df.to_csv(err_df_name, index=False)

    print('z_err: ', df['z_err'].describe())
    print('xy_err: ', df['xy_err'].describe())
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    sns.histplot(data=df['z_err'],)
    plt.title(f'{series_description} oof z_err')
    plt.savefig(f"{save_prefix}_z_err.png")
    plt.figure(figsize=(10, 10))
    sns.histplot(data=df['xy_err'],)
    plt.title(f'{series_description} oof xy_err')
    plt.savefig(f"{save_prefix}_xy_err.png")
    hard_z_list = []
    hard_xy_list = []
    for _, row in df.iterrows():
        study_id = int(row['study_id'])
        xy_err = row['xy_err']
        z_err = row['z_err']
        if z_err > 2:
            hard_z_list.append(study_id)
            #print('study_id: ', int(study_id), z_err)
        if xy_err > 10:
            hard_xy_list.append(study_id)
    print(len(hard_z_list))
    print('hard_z_list: ', hard_z_list)
    print(len(hard_xy_list))
    print('hard_xy_list: ', hard_xy_list)
