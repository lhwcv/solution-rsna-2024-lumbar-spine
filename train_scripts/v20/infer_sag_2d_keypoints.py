# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import DataLoader
import tqdm
import cv2
import numpy as np
from train_scripts.data_path import DATA_ROOT
from train_scripts.v20.models import RSNA24Model_Keypoint_2D
from infer_scripts.dataset import RSNA24DatasetTest_LHW_V2
import pickle


# def infer_keypoints(model, img):
#     s_t1 = img[:, :10]
#     s_t1 = s_t1[:, 4:7] # take the center
#     s_t2 = img[:, 10: 20]
#     s_t2 = s_t2[:, 4: 7]
#     s = torch.cat((s_t1, s_t2), dim=0)
#     with torch.no_grad():
#         pred = model(s).reshape(2, bs, 5, 2).permute(1, 0, 2, 3).cpu().numpy()
#     return pred

def load_models(model_dir='', model_name='densenet161'):
    model_fns = [
        f'{model_dir}/best_fold_0_ema.pt',
        f'{model_dir}/best_fold_1_ema.pt',
        f'{model_dir}/best_fold_2_ema.pt',
        f'{model_dir}/best_fold_3_ema.pt',
        f'{model_dir}/best_fold_4_ema.pt',
    ]
    models = []
    for i, fn in enumerate(model_fns):
        print('load from: ', fn)
        model = RSNA24Model_Keypoint_2D(model_name=model_name, num_classes=10).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models.append(model)
    return models

autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)

def infer_xy(models, img):
    # img: b, 1, h, w
    pts = None
    for i in range(len(models)):
        with torch.no_grad():
            with autocast:
                p = models[i](img)
                if pts is None:
                    pts = p
                else:
                    pts += p
    pts = pts / len(models)
    return pts


if __name__ == '__main__':
    data_root = DATA_ROOT
    df = pd.read_csv(f'{data_root}/train_series_descriptions.csv')
    study_ids = list(df['study_id'].unique())
    debug_dir = f'{data_root}/debug_dir/'
    from src.utils.comm import create_dir
    create_dir(debug_dir)

    save_dir = f'{data_root}/pred_keypoints/'
    create_dir(save_dir)

    transforms_val = A.Compose([
        A.Normalize(mean=0.5, std=0.5)
    ])

    dset = RSNA24DatasetTest_LHW_V2(data_root, study_ids,
                                    image_dir=data_root + '/train_images/',
                                    with_axial=False,
                                    test_series_descriptions_fn='train_series_descriptions.csv', )

    dloader = DataLoader(dset, batch_size=8, num_workers=8)

    #t1_models = load_models('./wkdir_final/keypoint_2d_v20_sag_t1/densenet161_lr_0.0006_sag_T1/')
    #t2_models = load_models('./wkdir_final/keypoint_2d_v20_sag_t2/densenet161_lr_0.0006_sag_T2/')

    t1_models = load_models('./wkdir_final/keypoint_2d_v20_sag_t1/fastvit_ma36.apple_dist_in1k_lr_0.0008_sag_T1/',
                            model_name='fastvit_ma36.apple_dist_in1k')
    t2_models = load_models('./wkdir_final/keypoint_2d_v20_sag_t2/convformer_s36.sail_in22k_ft_in1k_384_lr_0.0008_sag_T2/',
                            model_name='convformer_s36.sail_in22k_ft_in1k')

    study_id_to_pred_keypoints = {}

    for imgs, study_ids in tqdm.tqdm(dloader):
        bs, _, _, _ = imgs.shape
        imgs = imgs.cuda()
        s_t1 = imgs[:, :10]
        s_t1 = s_t1[:, 5:6]  # take the center
        s_t2 = imgs[:, 10: 20]
        s_t2 = s_t2[:, 5: 6]

        keypoints_t1 = infer_xy(t1_models, s_t1).reshape(bs, 1, 5, 2)
        keypoints_t2 = infer_xy(t2_models, s_t2).reshape(bs, 1, 5, 2)
        keypoints = torch.cat((keypoints_t1, keypoints_t2), dim=1).cpu().numpy()

        for idx, study_id in enumerate(study_ids):
            study_id = int(study_id)
            study_id_to_pred_keypoints[study_id] = keypoints[idx] * 512

        # imgs = imgs.cpu()
        # img = imgs[0, :10].numpy()
        # keypoints = keypoints[0]
        # for i in range(len(img)):
        #     for p in keypoints[0]:
        #         x, y = int(p[0] * 512), int(p[1] * 512)
        #         #print(x, y)
        #         img[i] = 255 * (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        #         img[i] = np.array(img[i], dtype=np.uint8)
        #         img[i] = cv2.circle(img[i].copy(), (x, y), 5, (255, 255, 255), 9)
        # img_concat = np.concatenate(img, axis=0)
        # cv2.imwrite(f'{save_dir}/x_t1.jpg', img_concat)
        #
        # img = imgs[0, 10: 20].numpy()
        # for i in range(len(img)):
        #     for p in keypoints[1]:
        #         x, y = int(p[0] * 512), int(p[1] * 512)
        #         #print(x, y)
        #         img[i] = 255 * (img[i] - img[i].min()) / (img[i].max() - img[i].min())
        #         img[i] = np.array(img[i], dtype=np.uint8)
        #         img[i] = cv2.circle(img[i].copy(), (x, y), 5, (255, 255, 255), 9)
        # img_concat = np.concatenate(img, axis=0)
        # cv2.imwrite(f'{save_dir}/x_t2.jpg', img_concat)
        # exit(0)

    # pickle.dump(study_id_to_pred_keypoints,
    #             open(save_dir + f'/v20_sag_2d_keypoints_center_slice.pkl', 'wb'))

    # pickle.dump(study_id_to_pred_keypoints,
    #             open(save_dir + f'/v20_sag_2d_keypoints_center_slice_f0.pkl', 'wb'))
    pickle.dump(study_id_to_pred_keypoints,
                open(save_dir + f'/v20_sag_2d_keypoints_center_slice_v2.pkl', 'wb'))

