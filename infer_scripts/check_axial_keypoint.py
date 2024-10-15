# -*- coding: utf-8 -*-
import glob
import pickle

import cv2
import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import KFold

from src.data.classification.v2_all_view import gen_level_group_idxs
from src.utils.aux_info import get_train_study_aux_info

_level_to_idx = {
    'L1/L2': 0,
    'L2/L3': 1,
    'L3/L4': 2,
    'L4/L5': 3,
    'L5/S1': 4
}


def load_volume(fns, index, bbox=None, img_size=128):
    arr = []
    if index is None:
        index = range(len(fns))
    for i in index:
        a = cv2.imread(fns[i], 0)
        if bbox is not None:
            a = a[bbox[1]: bbox[3], bbox[0]: bbox[2]].copy()
            try:
                a = cv2.resize(a, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                print(e)
                print(bbox)
                print(fns)
        arr.append(a)
    arr = np.array(arr, np.uint8)
    return arr





def predict_level_by_query(instance_num, group_idxs):
    for i, g in enumerate(group_idxs):
        if instance_num >= g[0] and instance_num <= g[1]:
            return i
    return -1


if __name__ == '__main__':

    DEBUG = False

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    volume_data_root = f'{data_root}/train_images_preprocessed/'
    debug_dir = f'{data_root}/debug_dir/'
    pred_keypoints_infos0 = pickle.load(
        open(f'{data_root}/v2_axial_3d_keypoints_model0.pkl', 'rb'))
    # pred_keypoints_infos1 = pickle.load(
    #     open(f'{data_root}/v2_axial_3d_keypoints_model1.pkl', 'rb'))

    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    aux_info = get_train_study_aux_info(data_root)

    skf = KFold(n_splits=5, shuffle=True, random_state=8620)
    df['fold'] = [0] * len(df)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        df.loc[val_idx, 'fold'] = fold

    df = df[df['fold'] == 0]

    samples = []
    for i, row in df.iterrows():
        label = row[1:].values.astype(np.int64)
        study_id = row['study_id']
        label_info = aux_info[study_id]['aux_infos']
        dataframes = {
            'series_id': [],
            'instance_number': [],
            'x': [],
            'y': [],
            'level': [],
            'cond': [],
        }
        for cond, coord_labels in label_info.items():
            if cond not in ['Left Subarticular Stenosis', 'Right Subarticular Stenosis']:
                continue
            for a in coord_labels:
                dataframes['series_id'].append(a['series_id'])
                dataframes['instance_number'].append(a['instance_number'])
                dataframes['x'].append(a['x'])
                dataframes['y'].append(a['y'])
                dataframes['level'].append(a['level'])
                dataframes['cond'].append(cond)

        dataframes = pd.DataFrame(dataframes)
        for series_id in dataframes['series_id'].unique():
            sub_dfs = dataframes[dataframes['series_id'] == series_id]
            # convert g to list of dict
            g = sub_dfs.to_dict('records')

            samples.append({
                'study_id': study_id,
                'series_id': series_id,
                'labeled_keypoints': g,
                'label': label
            })

    img_size = 512
    correct_n = 0
    total_n = 0
    z_errors = []
    for item in samples:

        study_id = int(item['study_id'])
        series_id = int(item['series_id'])

        meta_fn = f'{volume_data_root}/{study_id}/{series_id}/meta_info_2.pkl'
        meta = pickle.load(open(meta_fn, 'rb'))
        instance_num_to_shape = meta['instance_num_to_shape']
        dicom_instance_numbers_to_idx = {}
        for i, ins in enumerate(meta['dicom_instance_numbers']):
            dicom_instance_numbers_to_idx[ins] = i

        keypoints_left = np.zeros((5, 2), dtype=np.float32)
        keypoints_right = np.zeros((5, 2), dtype=np.float32)
        keypoints_z_left = np.zeros((5, 1), dtype=np.float32)
        keypoints_z_right = np.zeros((5, 1), dtype=np.float32)
        instance_numbers = set()
        for a in item['labeled_keypoints']:
            instance_number = a['instance_number']
            instance_numbers.add(instance_number)
            x, y = a['x'], a['y']
            origin_w, origin_h = instance_num_to_shape[instance_number]
            x = x * img_size / origin_w
            y = y * img_size / origin_h
            # query to current
            # print('instance_number origin: ', instance_number)
            instance_number = dicom_instance_numbers_to_idx[instance_number]
            # print('instance_number now: ', instance_number)

            idx = _level_to_idx[a['level']]
            if a['cond'] == 'Left Subarticular Stenosis':
                keypoints_left[idx, 0] = int(x)
                keypoints_left[idx, 1] = int(y)
                keypoints_z_left[idx] = instance_number
            else:
                keypoints_right[idx, 0] = int(x)
                keypoints_right[idx, 1] = int(y)
                keypoints_z_right[idx] = instance_number

        keypoints = np.concatenate((keypoints_left, keypoints_right), axis=0)
        keypoints_z = np.concatenate((keypoints_z_left, keypoints_z_right), axis=0)
        keypoints = np.concatenate((keypoints, keypoints_z), axis=1)

        # predict
        pred_info0 = pred_keypoints_infos0[study_id][series_id]
        #pred_info1 = pred_keypoints_infos1[study_id][series_id]

        predict_keypoints = pred_info0['points']
        # predict_keypoints = (pred_info0['points'] + pred_info1['points']) /2
        depth = int(pred_info0['d'])

        scale = img_size / 4.0
        scale_z = depth / 16.0
        predict_keypoints[:, :2] = predict_keypoints[:, :2] * scale
        predict_keypoints[:, 2] = predict_keypoints[:, 2] * scale_z

        if DEBUG:
            fns = glob.glob(f'{volume_data_root}/{study_id}/{series_id}/*.png')
            fns = sorted(fns)
            volumn = load_volume(fns, None, bbox=None)
            imgs = []
            for p in keypoints:
                x, y, z = int(p[0]), int(p[1]), int(p[2])
                if x == 0:
                    continue
                img = volumn[z]

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
                cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                imgs.append(img)
            img_concat = np.concatenate(imgs, axis=0)
            cv2.imwrite(f'{debug_dir}/check_axial_keypoint_gt.jpg', img_concat)

            ##
            imgs = []
            for p in predict_keypoints:
                x, y, z = int(p[0]), int(p[1]), int(p[2])
                img = volumn[z]

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
                cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                imgs.append(img)
            img_concat = np.concatenate(imgs, axis=0)
            cv2.imwrite(f'{debug_dir}/check_axial_keypoint_pred.jpg', img_concat)

            exit(0)

        ##
        pred_z = predict_keypoints[:, 2].reshape(2, 5)
        # print(pred_z)
        pred_z = np.round(pred_z.mean(axis=0))
        # print(pred_z)

        group_idxs = gen_level_group_idxs(pred_z, depth, ext=1.0)
        # print(group_idxs)

        for a in item['labeled_keypoints']:
            instance_number = a['instance_number']
            instance_number = dicom_instance_numbers_to_idx[instance_number]
            pred_idx = predict_level_by_query(instance_number, group_idxs)
            gt_idx = _level_to_idx[a['level']]

            err = abs(pred_z[gt_idx] - instance_number)
            z_errors.append(err)
            # print('pred_idx:', pred_idx)
            # print('gt_idx:', gt_idx)
            total_n += 1
            if pred_idx == gt_idx:
                correct_n += 1

        # exit(0)
        # break
    print('total n: ', total_n)
    print('correct_n: ', correct_n)
    print('acc: ', correct_n / total_n)

    z_errors = pd.DataFrame(z_errors)
    print(z_errors.describe())
