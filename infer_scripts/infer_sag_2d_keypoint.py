# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import cv2
from infer_scripts.dataset import RSNA24DatasetTest_LHW_V2
import albumentations as A
from src.models.keypoint.model_2d_keypoint import RSNA24Model_Keypoint_2D
import torch
from torch.utils.data import Dataset,DataLoader
import tqdm

def infer_sag_keypoints(model, img):
    bs, _, h, w = imgs.shape

    s_t1 = img[:, :10]
    s_t2 = img[:, 10: 20]

    s_t1 = s_t1.reshape(bs * 10, 1, h, w)
    s_t2 = s_t2.reshape(bs * 10, 1, h, w)

    with torch.no_grad():
        pred_t1 = model(s_t1).reshape(bs, 1, 10, 5, 2)
        pred_t2 = model(s_t2).reshape(bs, 1, 10, 5, 2)

    pred = torch.cat((pred_t1, pred_t2), dim=1).cpu().numpy()
    return pred


if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train_series_descriptions.csv')
    study_ids = list(df['study_id'].unique())
    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir
    create_dir(debug_dir)

    dset = RSNA24DatasetTest_LHW_V2(data_root,study_ids,
                                    image_dir=data_root+'/train_images/',
                                    with_axial=False,
                                    test_series_descriptions_fn='train_series_descriptions.csv',)

    dloader = DataLoader(dset, batch_size=8, num_workers=8)


    model_fns = [
        '../train_scripts/wkdir/keypoint/sag_2d/densenet161_lr_0.0006/best_fold_0_ema.pt',
        '../train_scripts/wkdir/keypoint/sag_2d/densenet161_lr_0.0006/best_fold_1_ema.pt',
    ]
    for i, fn in enumerate(model_fns):
        print('load from: ', fn)
        model = RSNA24Model_Keypoint_2D(model_name='densenet161').cuda()
        model.load_state_dict(torch.load(fn))

        study_id_to_pred_keypoints = {}

        for imgs, study_ids in tqdm.tqdm(dloader):
            bs, _, _, _ = imgs.shape
            keypoints = infer_sag_keypoints(model, imgs.cuda())
            for idx, study_id in enumerate(study_ids):
                study_id = int(study_id)
                study_id_to_pred_keypoints[study_id] = keypoints[idx] * 128
        #
        #     img = imgs[0, :10].numpy()
        #     keypoints = keypoints[0]
        #     for i in range(len(img)):
        #         ps = keypoints[0][i]
        #         for p in ps:
        #             x, y = int(p[0] * 128), int(p[1] * 128)
        #             #print(x, y)
        #             img[i] = 255 * (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        #             img[i] = np.array(img[i], dtype=np.uint8)
        #             img[i] = cv2.circle(img[i].copy(), (x, y), 5, (255, 255, 255), 9)
        #     img_concat = np.concatenate(img, axis=0)
        #     cv2.imwrite(f'{debug_dir}/x_t1.jpg', img_concat)
        #
        #     img = imgs[0, 10: 20].numpy()
        #     for i in range(len(img)):
        #         ps = keypoints[1][i]
        #         for p in ps:
        #             x, y = int(p[0] * 128), int(p[1] * 128)
        #             #print(x, y)
        #             img[i] = 255 * (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        #             img[i] = np.array(img[i], dtype=np.uint8)
        #             img[i] = cv2.circle(img[i].copy(), (x, y), 5, (255, 255, 255), 9)
        #     img_concat = np.concatenate(img, axis=0)
        #     cv2.imwrite(f'{debug_dir}/x_t2.jpg', img_concat)
        #     exit(0)

        import pickle

        pickle.dump(study_id_to_pred_keypoints,
                    open(data_root + f'/v2_sag_2d_keypoints_model{i}.pkl', 'wb'))

