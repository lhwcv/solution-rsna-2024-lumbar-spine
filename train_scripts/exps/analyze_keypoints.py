# -*- coding: utf-8 -*-
import tqdm
import pickle
import pandas as pd
import numpy as np
from src.data.keypoint.sag_3d import RSNA24Dataset_Sag_Cls_Use_GT_Point, RSNA24Dataset_KeyPoint_Sag_3D
from src.utils.aux_info import get_train_study_aux_info

if __name__ == '__main__':
    pred_df1 = pd.read_csv('../xyz_use_gt_pred.csv')
    pred_df2 = pd.read_csv('../z_use_pred_pred.csv')
    pred_df3 = pd.read_csv('../xy_use_pred_pred.csv')

    pred_df = pred_df1.merge(pred_df2, on=['study_id']).merge(pred_df3, on=['study_id'])

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'  # data_root = '/root/autodl-tmp/data/'
    df = pd.read_csv(f'{data_root}/train.csv')

    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    aux_info = get_train_study_aux_info(data_root)

    # dset = RSNA24Dataset_KeyPoint_Sag_3D(data_root, aux_info, df, )
    # study_id_to_pred_keypoints = {}
    # for d in tqdm.tqdm(dset):
    #     study_id_to_pred_keypoints[d['study_ids']] = d['keypoints']
    #
    #
    # pickle.dump(study_id_to_pred_keypoints,
    #             open(data_root + f'/v2_sag_3d_keypoints_gt.pkl', 'wb'))
    #
    # exit(0)

    #dset = RSNA24Dataset_Sag_Cls_Use_GT_Point(data_root, aux_info, df, )
    #pred_keypoints = dset.pred_sag_keypoints_infos_3d

    pred_keypoints = pickle.load(
        open(f'{data_root}/v2_keypoint_3d_2d_cascade_0823.pkl', 'rb'))
    #print(pred_keypoints)

    # pred_keypoints = pickle.load(
    #     open(f'{data_root}/v2_sag_3d_keypoints_en3.pkl', 'rb'))
    #
    #
    # pred_sag_keypoints_3d_2d_cascade = pickle.load(
    #     open(f'{data_root}/v2_keypoint_3d_2d_cascade_0821.pkl', 'rb'))
    #
    # for study_id in pred_keypoints.keys():
    #     p = pred_keypoints[study_id]['points'].reshape(15, 3)
    #     d = pred_sag_keypoints_3d_2d_cascade[study_id]
    #
    #     t2_keypoints = d['t2_keypoints']  # 5, 5, 3
    #     t1_keypoints = d['t1_keypoints']  # 10, 5, 3
    #
    #     t2_keypoints[:, :, :2] = 4 * t2_keypoints[:, :, :2] / 512
    #     t1_keypoints[:, : ,:2] = 4 * t1_keypoints[:, :, :2] / 512
    #
    #     for i in range(5):
    #         p[i][:2] = t2_keypoints[i, i][:2]
    #     for i in range(5):
    #         p[i + 5][:2] = t1_keypoints[i, i][:2]
    #     for i in range(5):
    #         p[i + 10][:2] = t1_keypoints[i + 5, i][:2]
    #
    #     pred_keypoints[study_id]['points'] = p


    gt_keypoins = pickle.load(open(f'{data_root}/v2_sag_3d_keypoints_gt.pkl', 'rb'))


    def _masked_err(pred, gt):
        pred = pred.reshape(-1)
        gt = gt.reshape(-1)
        mask = np.where(gt < 0, 0, 1)
        err = np.abs(pred * mask - gt * mask).sum() / mask.sum()
        return err

    pred_df['err_xy_512'] = [0]* len(pred_df)
    pred_df['err_z_16'] = [0] * len(pred_df)
    pred_df['degrade_xy'] = [0] * len(pred_df)
    pred_df['degrade_z'] = [0] * len(pred_df)

    large_err_list = []
    for i, row in pred_df.iterrows():
        study_id = int(row['study_id'])
        pred = pred_keypoints[study_id]['points'].reshape(15, 3)
        gt = gt_keypoins[study_id].reshape(15, 3)

        left_side_z = gt[5:10, 2].mean()
        right_size_z = gt[10:15, 2].mean()
        # if left_side_z <= right_size_z:
        #     print('study_id: ', int(study_id))
        #     print(gt[5:10, 2],gt[10:15, 2])

        # pred_xy = pred[:, :2]
        # gt_xy = gt[:, :2]

        pred_xy = pred[:, 1]
        gt_xy = gt[:, 1]

        err_xy_512 = 512 / 4 * _masked_err(pred_xy, gt_xy)

        if err_xy_512 > 10:
            print('study_id: ', study_id)
            large_err_list.append(study_id)

        pred_z = pred[:, 2]
        gt_z = gt[:, 2]
        err_z_16 = _masked_err(pred_z, gt_z)

        degrade_xy = row['xy_use_pred_weight'] - row['xyz_use_gt_weight']
        degrade_z = row['z_use_pred_weight'] - row['xyz_use_gt_weight']

        pred_df.loc[pred_df.index[i], 'err_xy_512'] = err_xy_512
        pred_df.loc[pred_df.index[i], 'err_z_16'] = err_z_16
        pred_df.loc[pred_df.index[i], 'degrade_xy'] = degrade_xy
        pred_df.loc[pred_df.index[i], 'degrade_z'] = degrade_z

        # if err_z_16 > 1:
        #     print('study_id: ', int(study_id))
            #print('pred left_right: ', pred_z[5:].astype(np.int64))
            #print('gt left_right: ', gt_z[5:].astype(np.int64))
            #print('='*20)

    print('err_z_16：', pred_df['err_z_16'].describe())
    print('err_xy_512：', pred_df['err_xy_512'].describe())
    print('large_err_list: ', large_err_list)
    print('len: ', len(large_err_list))
    exit(0)
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    joint_plot = sns.jointplot(data=pred_df, x="err_xy_512", y="degrade_xy", kind="reg")
    # 保存图像到文件，dpi参数可以调整图像的分辨率
    joint_plot.savefig("degrade_xy.png", dpi=300)

    plt.figure(figsize=(10, 10))
    joint_plot = sns.jointplot(data=pred_df, x="err_z_16", y="degrade_z", kind="reg")
    # 保存图像到文件，dpi参数可以调整图像的分辨率
    joint_plot.savefig("degrade_z.png", dpi=300)

    print('xyz_use_gt_weight: ')
    print(pred_df['xyz_use_gt_weight'].mean())
    print('xy_use_pred_weight: ')
    print(pred_df['xy_use_pred_weight'].mean())
    print('z_use_pred_weight: ')
    print(pred_df['z_use_pred_weight'].mean())
    print('===='*10)
    part_df = pred_df[pred_df['err_xy_512'] > 10]
    print('err_xy_512>10: ', part_df['xy_use_pred_weight'].mean(), part_df['xyz_use_gt_weight'].mean())

    part_df = pred_df[pred_df['err_xy_512'] < 6]
    print('err_xy_512 < 6: ', part_df['xy_use_pred_weight'].mean(),part_df['xyz_use_gt_weight'].mean())

    part_df = pred_df[pred_df['err_z_16'] > 2]
    print('err_z_16>1: ', part_df['z_use_pred_weight'].mean(), part_df['xyz_use_gt_weight'].mean())

    part_df = pred_df[pred_df['err_z_16'] <= 1]
    print('err_z_16< 1: ', part_df['z_use_pred_weight'].mean(),part_df['xyz_use_gt_weight'].mean())

    part_df = pred_df[pred_df['err_xy_512'] < 8]
    part_df = part_df[part_df['err_z_16'] <=1]

    print(len(part_df))
    print('err_xy_512 < 6: ', part_df['xy_use_pred_weight'].mean(), part_df['xyz_use_gt_weight'].mean())

    #part_df['study_id'].to_csv('./easy_list_0813.csv', index=False)
