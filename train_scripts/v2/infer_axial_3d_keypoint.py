# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
from infer_scripts.dataset import RSNA24DatasetTest_LHW_keypoint_3D_Axial
import albumentations as A
from src.models.keypoint.model_3d_keypoint import RSNA24Model_Keypoint_3D_Axial
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

import pickle
from train_scripts.data_path import DATA_ROOT

if __name__ == '__main__':
    data_root = DATA_ROOT
    df = pd.read_csv(f'{data_root}/train_series_descriptions.csv')
    study_ids = list(df['study_id'].unique())

    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir
    create_dir(debug_dir)
    save_dir = f'{data_root}/pred_keypoints/'
    create_dir(save_dir)
    study_ids = study_ids[:8]
    dset = RSNA24DatasetTest_LHW_keypoint_3D_Axial(data_root,
                                                   'train_series_descriptions.csv',
                                                   study_ids,
                                                   image_dir=data_root + '/train_images/')

    dloader = DataLoader(dset, batch_size=8, num_workers=8)

    model_fns = [
        #'./wkdir/keypoint/axial_3d_fix/densenet161_lr_0.0006/best_fold_0_ema.pt',
        #'./wkdir/keypoint/axial_3d_fix/densenet161_lr_0.0006/best_fold_1_ema.pt',
        './wkdir_final/keypoint_3d_v2_axial/densenet161_lr_0.0006/best_fold_0_ema.pt',
        './wkdir_final/keypoint_3d_v2_axial/densenet161_lr_0.0006/best_fold_1_ema.pt',
        './wkdir_final/keypoint_3d_v2_axial/densenet161_lr_0.0006/best_fold_2_ema.pt',
    ]
    models = []

    for i, fn in enumerate(model_fns):
        print('load from: ', fn)
        model = RSNA24Model_Keypoint_3D_Axial("densenet161", pretrained=False)
        model.load_state_dict(torch.load(fn))
        model.cuda()
        model.eval()
        models.append(model)

    study_id_to_pred_keypoints = {}
    autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)

    for volumns, study_ids, sids, depths in tqdm.tqdm(dloader):
        bs, _, _, _ = volumns.shape
        keypoints = None
        with torch.no_grad():
            with autocast:
                for i in range(len(models)):
                    p = models[i](volumns.cuda()).cpu().numpy().reshape(bs, 10, 3)
                    if keypoints is None:
                        keypoints = p
                    else:
                        keypoints += p
        keypoints = keypoints / len(models)

        for idx, study_id in enumerate(study_ids):
            study_id = int(study_id)
            sid = int(sids[idx])
            d = depths[idx]
            if study_id not in study_id_to_pred_keypoints.keys():
                study_id_to_pred_keypoints[study_id] = {}
            study_id_to_pred_keypoints[study_id][sid] = {
                'points': keypoints[idx],
                'd': int(d)
            }

        # volumn = volumns[0]
        # keypoints = keypoints[0]
        # imgs = []
        # scale = volumn.shape[1] / 4
        # for p in keypoints:
        #     x, y, z = int(p[0] * scale), int(p[1] * scale), int(p[2] * volumn.shape[0] / 16)
        #     print('xyz: ', x, y, z)
        #     img = volumn[z]
        #     img = 255 * (img - img.min()) / (img.max() - img.min())
        #     img = np.array(img, dtype=np.uint8)
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #     img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
        #     cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #
        #     imgs.append(img)
        # img_concat = np.concatenate(imgs, axis=0)
        # cv2.imwrite(f'{save_dir}/v2_axial_3d_pred.jpg', img_concat)
        # exit(0)

    # pickle.dump(study_id_to_pred_keypoints,
    #             open(save_dir + f'/v2_axial_3d_keypoints_en3_folds8.pkl', 'wb'))
    # pickle.dump(study_id_to_pred_keypoints,
    #             open(save_dir + f'/v2_axial_3d_keypoints_f0_folds8.pkl', 'wb'))

    pred_keypoints_infos0 = pickle.load(
        open(f'{data_root}/pred_keypoints/v2_axial_3d_keypoints_en3_folds8.pkl', 'rb'))
    for study_id, items0 in study_id_to_pred_keypoints.items():
        if study_id in pred_keypoints_infos0:
            items1 = pred_keypoints_infos0[study_id]
            for sid, v in items0.items():
                v2 = items1[sid]
                err = np.mean(np.abs(v['points'] - v2['points']))
                print(err)
