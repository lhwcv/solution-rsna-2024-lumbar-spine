# -*- coding: utf-8 -*-
import os
# os.environ['OMP_NUM_THREADS'] = '1'
# import cv2
# cv2.setNumThreads(0)
from torch.multiprocessing import Pool
from concurrent.futures.thread import ThreadPoolExecutor

import tqdm

from infer_scripts.v2.dataset import RSNA24DatasetTest_LHW_keypoint_3D_Axial, RSNA24DatasetTest_LHW_V2, \
    RSNA24DatasetTest_LHW_keypoint_3D_Saggital
from infer_scripts.v2.models import RSNA24Model_Keypoint_3D_Sag_V2
from infer_scripts.v24.infer_utils import *
import argparse
import pickle

IS_KAGGLE = False
if IS_KAGGLE:
    test_series_descriptions_fn = 'test_series_descriptions.csv'
    image_dir = 'test_images'
    N_WORKERS = 4
    CACHE_DIR = './cache/'
else:
    test_series_descriptions_fn = 'train_series_descriptions.csv'
    image_dir = 'train_images'
    N_WORKERS = 16
    CACHE_DIR = './cache/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/')
    parser.add_argument('--model_dir', type=str, default="./wkdir_final/")
    parser.add_argument('--model_dir2', type=str, default="./wkdir_final2/")
    parser.add_argument('--save_dir', type=str, default="./keypoints_pred/")
    return parser.parse_args()

WITH_MORE_MODEL = True

def infer_axial_3d_keypoints_v2(data_root,
                                study_ids,
                                model_dir,
                                model_dir2,
                                device,
                                num_workers=8,
                                is_parallel=False):
    """
    this has moved to infer_axial_3d_keypoints_v24 for dataloader speed
    """
    model_cfgs = [
        {
            'sub_dir': 'keypoint_3d_v2_axial/densenet161_lr_0.0006',
            'backbone': 'densenet161',
        }
    ]
    v2_folds = 3
    models = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(v2_folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            model = RSNA24Model_Keypoint_3D(backbone,
                                            num_classes=30,
                                            pretrained=False)
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models.append(model)

    dset = RSNA24DatasetTest_LHW_keypoint_3D_Axial(data_root,
                                                   test_series_descriptions_fn,
                                                   study_ids,
                                                   image_dir=data_root + f'/{image_dir}/')

    dloader = DataLoader(dset, batch_size=8, num_workers=num_workers)

    study_id_to_pred_keypoints = {}
    for volumns, study_ids_, sids, depths in tqdm.tqdm(dloader,desc=f'{device}'):
        bs, _, _, _ = volumns.shape
        volumns = volumns.unsqueeze(1).to(device)
        keypoints = None
        with torch.no_grad():
            with autocast:
                for i in range(len(models)):
                    p = models[i](volumns).cpu().numpy().reshape(bs, 10, 3)
                    if keypoints is None:
                        keypoints = p
                    else:
                        keypoints += p
        keypoints = keypoints / len(models)
        for idx, study_id in enumerate(study_ids_):
            study_id = int(study_id)
            sid = int(sids[idx])
            d = depths[idx]
            # print(study_id, sid)
            if study_id not in study_id_to_pred_keypoints.keys():
                study_id_to_pred_keypoints[study_id] = {}
            study_id_to_pred_keypoints[study_id][sid] = {
                'points': keypoints[idx],
                'd': int(d)
            }
    return study_id_to_pred_keypoints


def infer_axial_3d_keypoints_v24(data_root,
                                 study_ids,
                                 model_dir,
                                 model_dir2,
                                 device,
                                 num_workers=8,
                                 is_parallel=False,
                                 ):
    z_model_cfgs = [
        {
            'sub_dir': 'keypoint_3d_v24_axial/level_cls/convnext_small.in12k_ft_in1k_384',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
        },
    ]
    z_folds = 1
    z_models = []
    for cfg in z_model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(z_folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            print('load: ', fn)
            model = Axial_Level_Cls_Model_for_Test(backbone,
                                                   pretrained=False).to(device)
            model.load_state_dict(torch.load(fn))
            model.eval()
            z_models.append(model)

    if WITH_MORE_MODEL:
        z_model_cfgs = [
            {
                'sub_dir': 'keypoint_3d_v24_axial/level_cls/pvt_v2_b2.in1k',
                'backbone': 'pvt_v2_b2.in1k',
            },
        ]
        z_models2 = []
        for cfg in z_model_cfgs:
            sub_dir = cfg['sub_dir']
            backbone = cfg['backbone']
            for fold in [1]:
                fn = os.path.join(model_dir2, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'
                print('load: ', fn)
                model = Axial_Level_Cls_Model_for_Test(backbone,
                                                       pretrained=False).to(device)
                model.load_state_dict(torch.load(fn))
                model.eval()
                z_models2.append(model)

    #
    xy_model_cfgs = [
        # {
        #     'sub_dir': 'keypoint_3d_v24_axial/axial_2d_keypoints/densenet161_lr_0.0006',
        #     'backbone': 'densenet161',
        # },
        {
            'sub_dir': 'keypoint_2d_v24_axial/xception65.tf_in1k_lr_0.0006/',
            'backbone': 'xception65.tf_in1k',
        },
    ]
    xy_folds = 5
    xy_models = []
    for cfg in xy_model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(xy_folds):
            fn = os.path.join(model_dir2, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            print('load: ', fn)
            model = RSNA24Model_Keypoint_2D(backbone,
                                            pretrained=False,
                                            num_classes=4,
                                            ).to(device)
            model.load_state_dict(torch.load(fn))
            model.eval()
            xy_models.append(model)

    v24_axial_pred_keypoints_info = {}
    v24_axial_pred_keypoints_info_2 = {}
    v2_axial_pred_keypoints_info = {}

    # v2 moved to here for dataloader speed
    boost_v2_dataloader = False
    if boost_v2_dataloader:
        model_cfgs = [
            {
                'sub_dir': 'keypoint_3d_v2_axial/densenet161_lr_0.0006',
                'backbone': 'densenet161',
            }
        ]
        v2_folds = 3
        v2_models = []
        for cfg in model_cfgs:
            sub_dir = cfg['sub_dir']
            backbone = cfg['backbone']
            for fold in range(v2_folds):
                fn = os.path.join(model_dir, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'
                model = RSNA24Model_Keypoint_3D(backbone,
                                                num_classes=30,
                                                pretrained=False).to(device)
                model.load_state_dict(torch.load(fn))
                model.eval()
                v2_models.append(model)

    dset = Axial_Level_Dataset_Multi_V24(data_root, study_ids,
                                         test_series_descriptions_fn=test_series_descriptions_fn,
                                         image_dir=image_dir)
    dloader = DataLoader(dset, batch_size=1,
                         num_workers=num_workers,
                         collate_fn=axial_v24_collate_fn)


    for data_dict in tqdm.tqdm(dloader,desc=f'{device}'):
        study_id, series_id_list, pred_keypoints = axial_v24_infer_z(z_models, data_dict, device)
        study_id = int(study_id)
        pred_keypoints = axial_v24_infer_xy(xy_models, pred_keypoints, data_dict)

        if boost_v2_dataloader:
            v2_pred = axial_v2_infer_xyz(v2_models, data_dict)
            v2_axial_pred_keypoints_info[study_id] = v2_pred

        v24_axial_pred_keypoints_info[study_id] = {}
        for i, sid in enumerate(series_id_list):
            sid = int(sid)
            v24_axial_pred_keypoints_info[study_id][sid] = {
                'points': pred_keypoints[i]  # 2, 5, 4
            }
        if WITH_MORE_MODEL:
            study_id, series_id_list, pred_keypoints = axial_v24_infer_z(z_models2, data_dict, device)
            study_id = int(study_id)
            pred_keypoints = axial_v24_infer_xy(xy_models, pred_keypoints, data_dict)

            v24_axial_pred_keypoints_info_2[study_id] = {}
            for i, sid in enumerate(series_id_list):
                sid = int(sid)
                v24_axial_pred_keypoints_info_2[study_id][sid] = {
                    'points': pred_keypoints[i]  # 2, 5, 4
                }


    if not boost_v2_dataloader:
        v2_axial_pred_keypoints_info = infer_axial_3d_keypoints_v2(data_root,
                                                                   study_ids,
                                                                   model_dir,
                                                                   model_dir2,
                                                                   device,
                                                                   num_workers,
                                                                   is_parallel)
    if WITH_MORE_MODEL:
        return  (v24_axial_pred_keypoints_info, v24_axial_pred_keypoints_info_2, v2_axial_pred_keypoints_info)
    return (v24_axial_pred_keypoints_info, v2_axial_pred_keypoints_info)


def _infer_sag_xy(models, img):
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


def infer_sag_2d_keypoints_v2(data_root,
                              study_ids,
                              model_dir,
                              model_dir2,
                              device,
                              num_workers=8,
                              is_parallel=False):
    model_cfgs_t1 = [
        {
            'sub_dir': 'keypoint_2d_v20_sag_t1/densenet161_lr_0.0006_sag_T1',
            'backbone': 'densenet161',
        },
        {
            'sub_dir': 'keypoint_2d_v20_sag_t1/fastvit_ma36.apple_dist_in1k_lr_0.0008_sag_T1',
            'backbone': 'fastvit_ma36.apple_dist_in1k',
        },

    ]
    v2_folds = 5
    models_t1 = []
    for cfg in model_cfgs_t1:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(v2_folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            if not os.path.exists(fn):
                fn = os.path.join(model_dir2, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'

            model = RSNA24Model_Keypoint_2D(backbone,
                                            num_classes=10,
                                            pretrained=False)
            print(f'{device} -->load: ',fn)
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models_t1.append(model)
    #
    model_cfgs_t2 = [
        {
            'sub_dir': 'keypoint_2d_v20_sag_t2/densenet161_lr_0.0006_sag_T2',
            'backbone': 'densenet161',
        },
        {
            'sub_dir': 'keypoint_2d_v20_sag_t2/convformer_s36.sail_in22k_ft_in1k_384_lr_0.0008_sag_T2',
            'backbone': 'convformer_s36.sail_in22k_ft_in1k_384',
        },
    ]
    v2_folds = 5
    models_t2 = []
    for cfg in model_cfgs_t2:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(v2_folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            if not os.path.exists(fn):
                fn = os.path.join(model_dir2, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'

            model = RSNA24Model_Keypoint_2D(backbone,
                                            num_classes=10,
                                            pretrained=False)
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models_t2.append(model)

    #
    dset = RSNA24DatasetTest_LHW_V2(data_root, study_ids,
                                    image_dir=data_root + f'/{image_dir}/',
                                    test_series_descriptions_fn=test_series_descriptions_fn,
                                    with_axial=False,
                                    cache_dir=CACHE_DIR)
    print(f'{device} build data done!')
    dloader = DataLoader(dset, batch_size=8, num_workers=num_workers)

    study_id_to_pred_keypoints = {}

    for tensor_dict in tqdm.tqdm(dloader, desc=f'{device}'):
        imgs = tensor_dict['img']
        study_ids = tensor_dict['study_id']
        bs, _, _, _ = imgs.shape
        imgs = imgs.to(device)
        s_t1 = imgs[:, :10]
        s_t1 = s_t1[:, 5:6]  # take the center
        s_t2 = imgs[:, 10: 20]
        s_t2 = s_t2[:, 5: 6]

        keypoints_t1 = _infer_sag_xy(models_t1, s_t1).reshape(bs, 1, 5, 2)
        keypoints_t2 = _infer_sag_xy(models_t2, s_t2).reshape(bs, 1, 5, 2)
        keypoints = torch.cat((keypoints_t1, keypoints_t2), dim=1).cpu().numpy()

        for idx, study_id in enumerate(study_ids):
            study_id = int(study_id)
            study_id_to_pred_keypoints[study_id] = keypoints[idx] * 512
    return study_id_to_pred_keypoints


def infer_sag_3d_keypoints_v2(data_root,
                              study_ids,
                              model_dir,
                              model_dir2,
                              device,
                              num_workers=8,
                              is_parallel=False,
                              ):
    model_cfgs = [
        {
            'sub_dir': 'keypoint_3d_v2_sag/densenet161_lr_0.0006/',
            'backbone': 'densenet161',
        }
    ]
    v2_folds = 5
    models = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(v2_folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            print(f'{device} -->load: ', fn)
            model = RSNA24Model_Keypoint_3D_Sag_V2(model_name=backbone,
                                                   pretrained=False)
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models.append(model)
    #
    dset = RSNA24DatasetTest_LHW_keypoint_3D_Saggital(data_root,
                                                      test_series_descriptions_fn,
                                                      study_ids,
                                                      image_dir=data_root + f'{image_dir}',
                                                      cache_dir=CACHE_DIR)

    dloader = DataLoader(dset, batch_size=8, num_workers=num_workers)

    study_id_to_pred_keypoints_sag = {}
    for volumns, study_ids_ in tqdm.tqdm(dloader, desc=f'{device}'):
        with torch.no_grad():
            volumns = volumns.to(device)
            with autocast:
                keypoints = None
                for i in range(len(models)):
                    y = models[i](volumns)
                    bs = y.shape[0]
                    if keypoints is None:
                        keypoints = y.cpu().numpy().reshape(bs, 3, 5, 3)
                    else:
                        keypoints += y.cpu().numpy().reshape(bs, 3, 5, 3)
                keypoints = keypoints / len(models)
        for idx, study_id in enumerate(study_ids_):
            study_id = int(study_id)
            study_id_to_pred_keypoints_sag[study_id] = {
                'points': keypoints[idx],
            }
    return study_id_to_pred_keypoints_sag


def infer_sag_3d_t1_keypoints_v20(data_root,
                                  study_ids,
                                  model_dir,
                                  model_dir2,
                                  device,
                                  num_workers=8,
                                  is_parallel=False):
    with_cascade = False
    if with_cascade:
        model_cfgs_t1_2d = [
            {
                'sub_dir': 'keypoint_2d_v20_sag_t1/densenet161_lr_0.0006_sag_T1',
                'backbone': 'densenet161',
            }
        ]
        v2_folds = 5
        models_t1_2d = []
        for cfg in model_cfgs_t1_2d:
            sub_dir = cfg['sub_dir']
            backbone = cfg['backbone']
            for fold in range(v2_folds):
                fn = os.path.join(model_dir, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'
                model = RSNA24Model_Keypoint_2D(backbone,
                                                num_classes=10,
                                                pretrained=False)
                model.load_state_dict(torch.load(fn))
                model.to(device)
                model.eval()
                if is_parallel:
                    model = nn.DataParallel(model)
                models_t1_2d.append(model)

    #
    pred_sag_keypoints_infos_3d_t1 = {}

    model_cfgs = [
        {
            'sub_dir': 'keypoint_3d_v20_sag_t1/densenet161_lr_0.0006_sag_T1',
            'backbone': 'densenet161',
        },
        {
            'sub_dir': 'keypoint_3d_v20_sag_t1/resnet34d_lr_0.0006_sag_T1',
            'backbone': 'resnet34d',
        }
    ]
    folds = 5
    models = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            if not os.path.exists(fn):
                fn = os.path.join(model_dir2, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'
            model = RSNA24Model_Keypoint_3D(backbone,
                                            in_chans=1,
                                            pretrained=False,
                                            num_classes=30).to(device)
            model.load_state_dict(torch.load(fn))
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models.append(model)

    dset = Sag_3D_Point_Dataset_V24(data_root,
                                    study_ids,
                                    test_series_descriptions_fn=test_series_descriptions_fn,
                                    image_dir=image_dir,
                                    series_description='Sagittal T1',
                                    with_origin_arr=True if with_cascade else False,
                                    cache_dir=CACHE_DIR
                                    )

    dloader = DataLoader(dset, batch_size=1 if with_cascade else 8,
                         num_workers=num_workers)

    for tensor_dict in tqdm.tqdm(dloader,desc=f'{device}'):
        with torch.no_grad():
            x = tensor_dict['imgs'].to(device)
            study_id_list = tensor_dict['study_id']
            series_id_list = tensor_dict['series_id']
            with autocast:
                y = None
                for i in range(len(models)):
                    p = models[i](x)
                    if y is None:
                        y = p
                    else:
                        y += p
                y = y / len(models)
            bs = y.shape[0]
            for b in range(bs):
                study_id = int(study_id_list[b])
                series_id = int(series_id_list[b])
                pts = y[b].reshape(-1, 5, 3).cpu().numpy()

                if with_cascade:
                    origin_depth = int(tensor_dict['origin_depth'][b])
                    origin_imgs = tensor_dict['origin_imgs'][b].to(device)
                    sub_imgs = []
                    for n in range(pts.shape[0]):
                        for level in range(5):
                            z = pts[n, level, 2] * origin_depth
                            z = int(np.round(z))
                            if z < 0:
                                z = 0
                            if z > origin_depth - 1:
                                z = origin_depth - 1

                            img = origin_imgs[z].unsqueeze(0).unsqueeze(0)
                            sub_imgs.append(img)
                    sub_imgs = torch.cat(sub_imgs, dim=0)  # 2*5, 1, 512, 512
                    xy_pred = _infer_sag_xy(models_t1_2d, sub_imgs).reshape(-1, 5, 5, 2).cpu().numpy()
                    for n in range(pts.shape[0]):
                        for level in range(5):
                            pts[n, level, :2] = xy_pred[n, level, level]

                if study_id not in pred_sag_keypoints_infos_3d_t1:
                    pred_sag_keypoints_infos_3d_t1[study_id] = {}
                pred_sag_keypoints_infos_3d_t1[study_id][series_id] = pts

    return pred_sag_keypoints_infos_3d_t1


def infer_sag_3d_t2_keypoints_v20(data_root,
                                  study_ids,
                                  model_dir,
                                  model_dir2,
                                  device,
                                  num_workers=8,
                                  is_parallel=False):
    with_cascade = False
    if with_cascade:
        model_cfgs_t1_2d = [
            {
                'sub_dir': 'keypoint_2d_v20_sag_t2/densenet161_lr_0.0006_sag_T2',
                'backbone': 'densenet161',
            }
        ]
        v2_folds = 5
        models_t2_2d = []
        for cfg in model_cfgs_t1_2d:
            sub_dir = cfg['sub_dir']
            backbone = cfg['backbone']
            for fold in range(v2_folds):
                fn = os.path.join(model_dir, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'
                model = RSNA24Model_Keypoint_2D(backbone,
                                                num_classes=10,
                                                pretrained=False)
                model.load_state_dict(torch.load(fn))
                model.to(device)
                model.eval()
                if is_parallel:
                    model = nn.DataParallel(model)
                models_t2_2d.append(model)

    #
    pred_sag_keypoints_infos_3d_t2 = {}

    model_cfgs = [
        {
            'sub_dir': 'keypoint_3d_v20_sag_t2/densenet161_lr_0.0006_sag_T2',
            'backbone': 'densenet161',
        },
        {
            'sub_dir': 'keypoint_3d_v20_sag_t2/resnet34d_lr_0.0006_sag_T2',
            'backbone': 'resnet34d',
        }
    ]
    folds = 5
    models = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        for fold in range(folds):
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_fold_{fold}_ema.pt'
            if not os.path.exists(fn):
                fn = os.path.join(model_dir2, sub_dir)
                fn = fn + f'/best_fold_{fold}_ema.pt'
            model = RSNA24Model_Keypoint_3D(backbone,
                                            in_chans=1,
                                            pretrained=False,
                                            num_classes=15).to(device)
            model.load_state_dict(torch.load(fn))
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models.append(model)

    dset = Sag_3D_Point_Dataset_V24(data_root,
                                    study_ids,
                                    test_series_descriptions_fn=test_series_descriptions_fn,
                                    image_dir=image_dir,
                                    series_description='Sagittal T2/STIR',
                                    cache_dir=CACHE_DIR,
                                    with_origin_arr=True if with_cascade else False
                                    )

    dloader = DataLoader(dset, batch_size=1 if with_cascade else 8,
                         num_workers=num_workers)

    for tensor_dict in tqdm.tqdm(dloader,desc=f'{device}'):
        with torch.no_grad():
            x = tensor_dict['imgs'].to(device)
            study_id_list = tensor_dict['study_id']
            series_id_list = tensor_dict['series_id']
            with autocast:
                y = None
                for i in range(len(models)):
                    p = models[i](x)
                    if y is None:
                        y = p
                    else:
                        y += p
                y = y / len(models)
            bs = y.shape[0]
            for b in range(bs):
                study_id = int(study_id_list[b])
                series_id = int(series_id_list[b])
                pts = y[b].reshape(-1, 5, 3).cpu().numpy()

                if with_cascade:
                    origin_depth = int(tensor_dict['origin_depth'][b])
                    origin_imgs = tensor_dict['origin_imgs'][b].to(device)
                    sub_imgs = []
                    for n in range(pts.shape[0]):
                        for level in range(5):
                            z = pts[n, level, 2] * origin_depth
                            z = int(np.round(z))
                            if z < 0:
                                z = 0
                            if z > origin_depth - 1:
                                z = origin_depth - 1

                            img = origin_imgs[z].unsqueeze(0).unsqueeze(0)
                            sub_imgs.append(img)
                    sub_imgs = torch.cat(sub_imgs, dim=0)  # 2*5, 1, 512, 512
                    xy_pred = _infer_sag_xy(models_t2_2d, sub_imgs).reshape(-1, 5, 5, 2).cpu().numpy()
                    for n in range(pts.shape[0]):
                        for level in range(5):
                            pts[n, level, :2] = xy_pred[n, level, level]

                if study_id not in pred_sag_keypoints_infos_3d_t2:
                    pred_sag_keypoints_infos_3d_t2[study_id] = {}
                pred_sag_keypoints_infos_3d_t2[study_id][series_id] = pts

    return pred_sag_keypoints_infos_3d_t2


def infer_all_keypoints(data_root,
                        model_dir,
                        model_dir2,
                        with_v2_sag_center_slice_2d=True,
                        with_v2_axial_3d=True,
                        with_v2_sag_3d=True,
                        with_v20_sag_t1_3d=True,
                        with_v20_sag_t2_3d=True,
                        with_v24_axial_3d=True
                        ):
    print('with_v2_sag_center_slice_2d: ', with_v2_sag_center_slice_2d)
    print('with_v2_axial_3d: ', with_v2_axial_3d)
    print('with_v2_sag_3d: ', with_v2_sag_3d)
    print('with_v20_sag_t1_3d: ', with_v20_sag_t1_3d)
    print('with_v20_sag_t2_3d: ', with_v20_sag_t2_3d)
    print('with_v24_axial_3d: ', with_v24_axial_3d)

    df = pd.read_csv(f'{data_root}/{test_series_descriptions_fn}')
    study_ids = df['study_id'].unique().tolist()
    # study_ids = study_ids[:16]

    ret_dict = {}
    device_0 = torch.device('cuda:0')


    is_parallel = torch.cuda.device_count() > 1 and len(study_ids) >=2
    print('is_parallel: ', is_parallel)
    if with_v2_axial_3d or with_v24_axial_3d:
        pred_dict_v24, pred_dict_v24_2,  pred_dict_v2 = \
            infer_axial_3d_keypoints_v24(data_root, study_ids, model_dir,model_dir2,
                                         device_0,N_WORKERS,is_parallel)

        ret_dict['axial_3d_keypoints_v2'] = pred_dict_v2
        ret_dict['axial_3d_keypoints_v24'] = pred_dict_v24
        ret_dict['axial_3d_keypoints_v24_2'] = pred_dict_v24_2


    if with_v2_sag_3d:
        print('infer_sag_3d_keypoints_v2...')
        pred_dict = infer_sag_3d_keypoints_v2(data_root, study_ids,
                                              model_dir, model_dir2, device_0, N_WORKERS,
                                              is_parallel)
        ret_dict['sag_3d_keypoints_v2'] = pred_dict


    if with_v2_sag_center_slice_2d:
        print('infer_sag_2d_keypoints_v2...')
        pred_dict = infer_sag_2d_keypoints_v2(data_root, study_ids,
                                              model_dir, model_dir2, device_0, N_WORKERS,
                                              is_parallel)
        ret_dict['sag_keypoints_v2'] = pred_dict


    if with_v20_sag_t1_3d:
        print('infer_sag_3d_t1_keypoints_v20...')
        pred_dict = infer_sag_3d_t1_keypoints_v20(data_root, study_ids,
                                                  model_dir, model_dir2, device_0, N_WORKERS,
                                                  is_parallel)
        ret_dict['sag_t1_3d_keypoints_v20'] = pred_dict

    if with_v20_sag_t2_3d:
        print('infer_sag_3d_t2_keypoints_v20..')
        pred_dict = infer_sag_3d_t2_keypoints_v20(data_root, study_ids,
                                                  model_dir, model_dir2, device_0, N_WORKERS,
                                                  is_parallel)
        ret_dict['sag_t2_3d_keypoints_v20'] = pred_dict

    return ret_dict


if __name__ == '__main__':
    args = get_args()
    data_root = args.data_root
    model_dir = args.model_dir
    model_dir2 = args.model_dir2
    save_dir = args.save_dir

    sag_keypoints_v2 = True
    axial_3d_keypoints_v2 = True
    sag_3d_keypoints_v2 = True

    sag_t1_3d_keypoints_v20 = True
    sag_t2_3d_keypoints_v20 = True
    axial_3d_keypoints_v24 = True

    sag_keypoints_v2_save_fn = f'{save_dir}/sag_keypoints_v2.pkl'
    axial_3d_keypoints_v2_save_fn = f'{save_dir}/axial_3d_keypoints_v2.pkl'
    sag_3d_keypoints_v2_save_fn = f'{save_dir}/sag_3d_keypoints_v2.pkl'
    sag_t1_3d_keypoints_v20_save_fn = f'{save_dir}/sag_t1_3d_keypoints_v20.pkl'
    sag_t2_3d_keypoints_v20_save_fn = f'{save_dir}/sag_t2_3d_keypoints_v20.pkl'
    axial_3d_keypoints_v24_fn = f'{save_dir}/axial_3d_keypoints_v24.pkl'

    if os.path.exists(sag_keypoints_v2_save_fn):
        sag_keypoints_v2 = False
    if os.path.exists(axial_3d_keypoints_v2_save_fn):
        axial_3d_keypoints_v2 = False
    if os.path.exists(sag_3d_keypoints_v2_save_fn):
        sag_3d_keypoints_v2 = False

    if os.path.exists(sag_t1_3d_keypoints_v20_save_fn):
        sag_t1_3d_keypoints_v20 = False
    if os.path.exists(sag_t2_3d_keypoints_v20_save_fn):
        sag_t2_3d_keypoints_v20 = False
    if os.path.exists(axial_3d_keypoints_v24_fn):
        axial_3d_keypoints_v24 = False

    ret_dict = infer_all_keypoints(data_root,
                                   model_dir,
                                   model_dir2,
                                   with_v2_sag_center_slice_2d=sag_keypoints_v2,
                                   with_v2_axial_3d=axial_3d_keypoints_v2,
                                   with_v2_sag_3d=sag_3d_keypoints_v2,
                                   with_v20_sag_t1_3d=sag_t1_3d_keypoints_v20,
                                   with_v20_sag_t2_3d=sag_t2_3d_keypoints_v20,
                                   with_v24_axial_3d=axial_3d_keypoints_v24, )

    for k, v in ret_dict.items():
        with open(f'{save_dir}/{k}.pkl', 'wb') as file_handle:
            pickle.dump(v, file_handle)
