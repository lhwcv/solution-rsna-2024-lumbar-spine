# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
from infer_scripts.dataset import  RSNA24DatasetTest_LHW_keypoint_3D_Axial
import albumentations as A
from src.models.keypoint.model_3d_keypoint import RSNA24Model_Keypoint_3D_Axial
import torch
from torch.utils.data import Dataset,DataLoader
import tqdm



if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train_series_descriptions.csv')
    study_ids = list(df['study_id'].unique())

    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir

    dset = RSNA24DatasetTest_LHW_keypoint_3D_Axial(data_root,
                                 'train_series_descriptions.csv',
                                 study_ids,
                                 image_dir=data_root+'/train_images/')

    dloader = DataLoader(dset, batch_size=8, num_workers=8)

    model_fns = [
        '../train_scripts/wkdir/keypoint/axial_3d_fix/densenet161_lr_0.0006/best_fold_0_ema.pt',
        '../train_scripts/wkdir/keypoint/axial_3d_fix/densenet161_lr_0.0006/best_fold_1_ema.pt',
    ]
    for i, fn in enumerate(model_fns):
        print('load from: ', fn)
        model = RSNA24Model_Keypoint_3D_Axial("densenet161", pretrained=False)
        model.load_state_dict(torch.load(fn))
        model.cuda()
        model.eval()

        create_dir(debug_dir)

        study_id_to_pred_keypoints = {}

        for volumns, study_ids, sids, depths in tqdm.tqdm(dloader):
            bs, _, _, _ = volumns.shape
            with torch.no_grad():
                keypoints = model(volumns.cuda()).cpu().numpy().reshape(bs, 10, 3)

            for idx, study_id in enumerate(study_ids):
                study_id = int(study_id)
                sid = int(sids[idx])
                d = depths[idx]
                #print(study_id, sid)
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
            # cv2.imwrite(f'{debug_dir}/v4_pred2.jpg', img_concat)
            # exit(0)

        import pickle

        pickle.dump(study_id_to_pred_keypoints,
                    open(data_root + f'/v2_axial_3d_keypoints_model{i}.pkl', 'wb'))
