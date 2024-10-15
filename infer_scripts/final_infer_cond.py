# -*- coding: utf-8 -*-
import tqdm
from sklearn.model_selection import KFold

from infer_scripts.infer_cascade2_3 import make_calc
from infer_scripts.v2.dataset import RSNA24DatasetTest_LHW_V2, v2_collate_fn
from infer_scripts.v24.infer_utils import *
from infer_scripts.v2.models import HybridModel_V2
from infer_scripts.v20.models import build_v20_sag_model

import argparse
import pickle

IS_KAGGLE = False
if IS_KAGGLE:
    test_series_descriptions_fn = 'test_series_descriptions.csv'
    image_dir = 'test_images'
    N_WORKERS = 4
else:
    test_series_descriptions_fn = 'train_series_descriptions.csv'
    image_dir = 'train_images'
    N_WORKERS = 16

batch_size = 8

keypoint_dir = './keypoints_pred/'
CACHE_DIR = './cache/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/')
    parser.add_argument('--model_dir', type=str, default="./wkdir_final/")
    parser.add_argument('--model_dir2', type=str, default="./wkdir_final2/")
    parser.add_argument('--save_dir', type=str, default="./")
    return parser.parse_args()


def infer_v2(data_root,
             study_ids,
             model_dir,
             save_dir,
             device,
             num_workers=8,
             folds=None,
             is_parallel=False):
    model_cfgs = [
        {
            'sub_dir': 'v2_cond/pvt_v2_b2.in1k_axial_pvt_v2_b2.in1k_axial_size_256_sag_size_128',
            'backbone_sag': 'pvt_v2_b2.in1k',
            'backbone_axial': 'pvt_v2_b2.in1k',
            'weight': 0.3469085736266617,
        },

        {
            'sub_dir': 'v2_cond/convnext_small.in12k_ft_in1k_384_axial_densenet161_axial_size_256_sag_size_128',
            'backbone_sag': 'convnext_small.in12k_ft_in1k_384',
            'backbone_axial': 'densenet161',
            'weight': 0.20715647998779071,
        },
        {
            'sub_dir': 'v2_cond/convnext_nano.in12k_ft_in1k_axial_convnext_nano.in12k_ft_in1k_axial_size_256_sag_size_128',
            'backbone_sag': 'convnext_nano.in12k_ft_in1k',
            'backbone_axial': 'convnext_nano.in12k_ft_in1k',
            'weight': 0.1927349675794976,
        },
        {
            'sub_dir': 'v2_cond/pvt_v2_b1.in1k_axial_pvt_v2_b1.in1k_axial_size_256_sag_size_128',
            'backbone_sag': 'pvt_v2_b1.in1k',
            'backbone_axial': 'pvt_v2_b1.in1k',
            'weight': 0.13677624328967222,
        },
        {
            'sub_dir': 'v2_cond/pvt_v2_b1.in1k_axial_densenet161_axial_size_256_sag_size_128',
            'backbone_sag': 'pvt_v2_b1.in1k',
            'backbone_axial': 'densenet161',
            'weight': 0.11642373551637777,
        },

    ]
    v2_folds = list(range(5))
    if folds is not None:
        v2_folds = folds

    models = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone_sag = cfg['backbone_sag']
        backbone_axial = cfg['backbone_axial']
        w = cfg['weight']
        for fold in v2_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_score.pt'
            print('load: ', fn)
            model = HybridModel_V2(backbone_sag,
                                   backbone_axial,
                                   pretrained=False)
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models.append(model)
            weights.append(w)

    weights_sum = np.sum(weights)
    #
    with open(f'{keypoint_dir}/sag_keypoints_v2.pkl', 'rb') as f:
        sag_keypoints_v2 = pickle.load(f)

    with open(f'{keypoint_dir}/axial_3d_keypoints_v2.pkl', 'rb') as f:
        axial_3d_keypoints_v2 = pickle.load(f)

    valid_ds = RSNA24DatasetTest_LHW_V2(data_root,
                                        study_ids,
                                        test_series_descriptions_fn,
                                        study_id_to_pred_keypoints_sag=sag_keypoints_v2,
                                        study_id_to_pred_keypoints_axial=axial_3d_keypoints_v2,
                                        image_dir=data_root + f'{image_dir}/',
                                        cache_dir=CACHE_DIR)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=4,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=v2_collate_fn
    )

    all_preds = {}
    for tensor_d in tqdm.tqdm(valid_dl):
        with torch.no_grad():
            x = tensor_d['img'].to(device)
            axial_imgs = tensor_d['axial_imgs'].to(device)
            n_list = tensor_d['n_list']
            cond = tensor_d['cond'].to(device)
            study_id_list = tensor_d['study_id']
            preds = None
            with autocast:
                for i in range(len(models)):
                    p = models[i](x, axial_imgs, cond, n_list)
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, 5_cond, 5_level, 3
            preds = preds.reshape(bs, 5, 5, 3).cpu()
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds[study_id] = preds[b]

    return all_preds


def infer_v2_seed666(data_root,
                     study_ids,
                     model_dir,
                     save_dir,
                     device,
                     num_workers=8,
                     folds=None,
                     is_parallel=False):
    model_cfgs = [
        {
            'sub_dir': 'v2_cond/pvt_v2_b1.in1k_axial_pvt_v2_b1.in1k_axial_size_256_sag_size_128',
            'backbone_sag': 'pvt_v2_b1.in1k',
            'backbone_axial': 'pvt_v2_b1.in1k',
            'weight': 0.34092462492421793,
        },
        {
            'sub_dir': 'v2_cond/pvt_v2_b1.in1k_axial_densenet161_axial_size_256_sag_size_128',
            'backbone_sag': 'pvt_v2_b1.in1k',
            'backbone_axial': 'densenet161',
            'weight': 0.2458979925774557,
        },
        {
            'sub_dir': 'v2_cond/pvt_v2_b2.in1k_axial_pvt_v2_b2.in1k_axial_size_256_sag_size_128',
            'backbone_sag': 'pvt_v2_b2.in1k',
            'backbone_axial': 'pvt_v2_b2.in1k',
            'weight': 0.21641664109925535,
        },
        {
            'sub_dir': 'v2_cond/convnext_nano.in12k_ft_in1k_axial_convnext_nano.in12k_ft_in1k_axial_size_256_sag_size_128',
            'backbone_sag': 'convnext_nano.in12k_ft_in1k',
            'backbone_axial': 'convnext_nano.in12k_ft_in1k',
            'weight': 0.19676074139907107,
        },

    ]
    v2_folds = list(range(5))
    if folds is not None:
        v2_folds = folds

    models = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone_sag = cfg['backbone_sag']
        backbone_axial = cfg['backbone_axial']
        w = cfg['weight']
        for fold in v2_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_score.pt'
            print('load: ', fn)
            model = HybridModel_V2(backbone_sag,
                                   backbone_axial,
                                   pretrained=False,
                                   fea_dim=512)
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)
            models.append(model)
            weights.append(w)

    weights_sum = np.sum(weights)
    #
    with open(f'{keypoint_dir}/sag_keypoints_v2.pkl', 'rb') as f:
        sag_keypoints_v2 = pickle.load(f)

    with open(f'{keypoint_dir}/axial_3d_keypoints_v2.pkl', 'rb') as f:
        axial_3d_keypoints_v2 = pickle.load(f)
    #
    # with open(f'{keypoint_dir}/sag_t1_3d_keypoints_v20.pkl', 'rb') as f:
    #     pred_sag_keypoints_infos_3d_t1 = pickle.load(f)

    valid_ds = RSNA24DatasetTest_LHW_V2(data_root,
                                        study_ids,
                                        test_series_descriptions_fn,
                                        study_id_to_pred_keypoints_sag=sag_keypoints_v2,
                                        study_id_to_pred_keypoints_axial=axial_3d_keypoints_v2,
                                        # pred_sag_keypoints_infos_3d_t1=pred_sag_keypoints_infos_3d_t1,
                                        image_dir=data_root + f'{image_dir}/',
                                        cache_dir=CACHE_DIR)
    valid_dl = DataLoader(
        valid_ds,
        batch_size=4,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=v2_collate_fn
    )

    all_preds = {}
    for tensor_d in tqdm.tqdm(valid_dl):
        with torch.no_grad():
            x = tensor_d['img'].to(device)
            axial_imgs = tensor_d['axial_imgs'].to(device)
            n_list = tensor_d['n_list']
            cond = tensor_d['cond'].to(device)
            study_id_list = tensor_d['study_id']
            preds = None
            with autocast:
                for i in range(len(models)):
                    p = models[i](x, axial_imgs, cond, n_list)
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, 5_cond, 5_level, 3
            preds = preds.reshape(bs, 5, 5, 3).cpu()
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds[study_id] = preds[b]

    return all_preds


def infer_v24(  # df_valid,
        data_root,
        study_ids,
        model_dir,
        save_dir,
        device,
        num_workers=8,
        folds=None,
        is_parallel=False,
):
    def _forward_tta(model, x, cond):
        # return model(x, cond)
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x0 = model(x, cond)
        x1 = model(x_flip, cond)
        return (x0 + x1) / 2

    # t1

    model_cfgs = [
        {
            'sub_dir': 'v20_cond_t1/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_legacy',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': True,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': False,
            'data_key': '3_128_128',
            'weight': 0.20538912558481712,
        },
        {
            'sub_dir': 'v20_cond_t1/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h72_w128',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': False,
            'data_key': '3_72_128',
            'weight': 0.18552050200612183,
        },
        {
            'sub_dir': 'v20_cond_t1/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h72_w128_with_gru',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_72_128',
            'weight': 0.306110664432234,
        },
        {
            'sub_dir': 'v20_cond_t1/convnext_tiny.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru',
            'backbone': 'convnext_tiny.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_128_128',
            'weight': 0.30297970797682705,
        }

    ]
    v24_folds = list(range(5))
    if folds is not None:
        v24_folds = folds

    models = []
    data_keys = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        with_cond = cfg['with_cond']
        with_gru = cfg['with_gru']
        with_level_lstm = cfg['with_level_lstm']
        z_imgs = cfg['z_imgs']
        data_key = cfg['data_key']
        w = cfg['weight']
        for fold in v24_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_ema.pt'
            print('load: ', fn)
            model = build_v20_sag_model(
                backbone,
                with_cond=with_cond,
                with_gru=with_gru,
                with_level_lstm=with_level_lstm,
                z_imgs=z_imgs,
                pretrained=False
            )
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)

            models.append(model)
            weights.append(w)
            data_keys.append(data_key)
    weights_sum = np.sum(weights)
    with open(f'{keypoint_dir}/sag_t1_3d_keypoints_v20.pkl', 'rb') as f:
        pred_sag_keypoints_infos_3d_t1 = pickle.load(f)

    other_crop_size_list = [
        (3, 72, 128),
    ]
    dset = Sag_T1_Dataset_V24(data_root,
                              study_ids,
                              pred_sag_keypoints_infos_3d_t1,
                              test_series_descriptions_fn=test_series_descriptions_fn,
                              image_dir=image_dir,
                              crop_size_h=128,
                              crop_size_w=128,
                              cache_dir=CACHE_DIR,
                              other_crop_size_list=other_crop_size_list
                              )
    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
    )
    all_preds_t1 = {}
    sag_t1_preds = []
    for tensor_d in tqdm.tqdm(dloader):
        with torch.no_grad():
            for k in tensor_d.keys():
                if k not in ['study_id']:
                    tensor_d[k] = tensor_d[k].to(device)
            study_id_list = tensor_d['study_id']
            cond = tensor_d['cond']
            preds = None
            with autocast:
                for i in range(len(models)):
                    p = _forward_tta(models[i], tensor_d[data_keys[i]], cond)
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, n_cond, 5_level, 3
            preds = preds.reshape(bs, -1, 5, 3).cpu()
            sag_t1_preds.append(preds)
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds_t1[study_id] = preds[b]
    sag_t1_preds = torch.cat(sag_t1_preds, dim=0)

    # t2
    model_cfgs = [
        {
            'sub_dir': 'v20_cond_t2/rexnetr_200.sw_in12k_ft_in1k_z_imgs_3_seed_8620_h128_w128_level_lstm',
            'backbone': 'rexnetr_200.sw_in12k_ft_in1k',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '3_128_128',
            'weight': 0.2523387488397007,
        },

        {
            'sub_dir': 'v20_cond_t2/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_128_128',
            'weight': 0.24855049213914635,
        },
        {
            'sub_dir': 'v20_cond_t2/pvt_v2_b5.in1k_z_imgs_5_seed_8620_h128_w128_level_lstm',
            'backbone': 'pvt_v2_b5.in1k',
            'with_cond': False,
            'z_imgs': 5,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '5_128_128',
            'weight': 0.19532993129183054,
        },
        {
            'sub_dir': 'v20_cond_t2/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w96_level_lstm',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '3_64_96',
            'weight': 0.14061503567169062,
        },
        {
            'sub_dir': 'v20_cond_t2/convnext_tiny.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w96_level_lstm',
            'backbone': 'convnext_tiny.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '3_64_96',
            'weight': 0.090578570264412,
        },

        {
            'sub_dir': 'v20_cond_t2/convnext_tiny.in12k_ft_in1k_384_z_imgs_5_seed_8620_h128_w128_level_lstm',
            'backbone': 'convnext_tiny.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 5,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '5_128_128',
            'weight': 0.0725872217932197,
        },
    ]
    v24_folds = list(range(5))
    if folds is not None:
        v24_folds = folds

    models = []
    data_keys = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        with_cond = cfg['with_cond']
        with_gru = cfg['with_gru']
        with_level_lstm = cfg['with_level_lstm']
        z_imgs = cfg['z_imgs']
        data_key = cfg['data_key']
        w = cfg['weight']
        for fold in v24_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_ema.pt'
            print('load: ', fn)
            model = build_v20_sag_model(
                backbone,
                with_cond=with_cond,
                with_gru=with_gru,
                with_level_lstm=with_level_lstm,
                z_imgs=z_imgs,
                pretrained=False
            )
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)

            models.append(model)
            weights.append(w)
            data_keys.append(data_key)
    weights_sum = np.sum(weights)

    with open(f'{keypoint_dir}/sag_t2_3d_keypoints_v20.pkl', 'rb') as f:
        pred_sag_keypoints_infos_3d_t2 = pickle.load(f)

    with open(f'{keypoint_dir}/sag_3d_keypoints_v2.pkl', 'rb') as f:
        pred_sag_keypoints_infos_3d = pickle.load(f)

    for study_id in pred_sag_keypoints_infos_3d_t2.keys():
        pred_keypoints = pred_sag_keypoints_infos_3d[study_id]['points'].reshape(15, 3)
        t2_pred_keypoints = pred_keypoints[:5]
        t2_pred_keypoints[:, :2] = t2_pred_keypoints[:, :2] / 4.0
        t2_pred_keypoints[:, 2] = t2_pred_keypoints[:, 2] / 16.0
        for sid in pred_sag_keypoints_infos_3d_t2[study_id].keys():
            pred_sag_keypoints_infos_3d_t2[study_id][sid] = t2_pred_keypoints

    other_crop_size_list = [
        (3, 64, 96),
        (5, 128, 128),
    ]
    # other_crop_size_list = [
    # ]
    # aux_info = get_train_study_aux_info(data_root)
    # transforms_val = A.Compose([
    #     # A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    #     A.Normalize(mean=0.5, std=0.5)
    # ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    #
    # dset = RSNA24Dataset_Sag_Axial_Cls(
    #     data_root,
    #     aux_info,
    #     df_valid, phase='valid', transform=transforms_val,
    #     z_imgs=3,
    #     with_axial=False,
    #     crop_size_h=128,
    #     crop_size_w=128,
    #     resize_to_size=128,
    # )

    dset = Sag_T2_Dataset_V24(data_root,
                              study_ids,
                              pred_sag_keypoints_infos_3d_t2,
                              test_series_descriptions_fn=test_series_descriptions_fn,
                              image_dir=image_dir,
                              crop_size_h=128,
                              crop_size_w=128,
                              cache_dir=CACHE_DIR,
                              other_crop_size_list=other_crop_size_list
                              )
    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
    )
    all_preds_t2 = {}
    sag_t2_preds = []
    for tensor_d in tqdm.tqdm(dloader):
        with torch.no_grad():
            for k in tensor_d.keys():
                if k not in ['study_id']:
                    tensor_d[k] = tensor_d[k].to(device)
            study_id_list = tensor_d['study_id']
            cond = None  # tensor_d['cond']
            preds = None
            with autocast:
                for i in range(len(models)):
                    # p = _forward_tta(models[i], tensor_d['img'], cond)
                    p = _forward_tta(models[i], tensor_d[data_keys[i]], cond)
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, n_cond, 5_level, 3
            preds = preds.reshape(bs, -1, 5, 3).cpu()
            sag_t2_preds.append(preds)
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds_t2[study_id] = preds[b]
    sag_t2_preds = torch.cat(sag_t2_preds, dim=0)
    # return all_preds_t2

    # axial
    model_cfgs = [
        {
            'sub_dir': 'v24_cond_axial/convnext_small.in12k_ft_in1k_384_z_imgs_3',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'axial_in_channels': 1,
            'weight': 0.19731044421522836,
            'z_imgs': 3,
            'data_key': 'axial_imgs_3_128_128'
        },
        {
            'sub_dir': 'v24_cond_axial/convnext_tiny.in12k_ft_in1k_384_z_imgs_3',
            'backbone': 'convnext_tiny.in12k_ft_in1k',
            'axial_in_channels': 1,
            'weight': 0.30013472174512595,
            'z_imgs': 3,
            'data_key': 'axial_imgs_3_128_128'
        },
        {
            'sub_dir': 'v24_cond_axial/densenet161_z_imgs_3',
            'backbone': 'densenet161',
            'axial_in_channels': 1,
            'weight': 0.1627137568736331,
            'z_imgs': 3,
            'data_key': 'axial_imgs_3_128_128'
        },
        {
            'sub_dir': 'v24_cond_axial/pvt_v2_b1.in1k_z_imgs_3',
            'backbone': 'pvt_v2_b1.in1k',
            'axial_in_channels': 1,
            'weight': 1.0,
            'z_imgs': 0.0987451411551922,
            'data_key': 'axial_imgs_3_128_128'
        },
        {
            'sub_dir': 'v24_cond_axial/rexnetr_200.sw_in12k_ft_in1k_z_imgs_5',
            'backbone': 'rexnetr_200.sw_in12k_ft_in1k',
            'axial_in_channels': 3,
            'weight': 0.2410959360108204,
            'z_imgs': 5,
            'data_key': 'axial_imgs_5_128_128'
        },

    ]
    v24_folds = list(range(5))
    if folds is not None:
        v24_folds = folds

    models = []
    data_keys = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        data_key = cfg['data_key']
        axial_in_channels = cfg['axial_in_channels']
        w = cfg['weight']
        for fold in v24_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_ema.pt'
            print('load: ', fn)
            model = Axial_HybridModel_24(backbone,
                                         backbone,
                                         pretrained=False,
                                         axial_in_channels=axial_in_channels
                                         )
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)

            models.append(model)
            weights.append(w)
            data_keys.append(data_key)
    weights_sum = np.sum(weights)

    with open(f'{keypoint_dir}/axial_3d_keypoints_v24.pkl', 'rb') as f:
        axial_3d_keypoints_v24 = pickle.load(f)

    other_crop_size_list = [
        (5, 128, 128),
    ]

    dset = Axial_Cond_Dataset_Multi_V24(data_root,
                                        study_ids,
                                        axial_3d_keypoints_v24,
                                        pred_sag_keypoints_infos_3d_t2,
                                        test_series_descriptions_fn=test_series_descriptions_fn,
                                        image_dir=image_dir,
                                        z_imgs=3,
                                        other_z_imgs_list=other_crop_size_list,
                                        cache_dir=CACHE_DIR,
                                        )

    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
    )
    all_preds_axial = {}
    axial_preds = []
    for tensor_d in tqdm.tqdm(dloader):
        with torch.no_grad():
            for k in tensor_d.keys():
                if k not in ['study_id']:
                    tensor_d[k] = tensor_d[k].to(device)
            study_id_list = tensor_d['study_id']
            preds = None
            with autocast:
                for i in range(len(models)):
                    p = models[i](tensor_d['sag_t2'], tensor_d[data_keys[i]])
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, n_cond, 5_level, 3
            preds = preds.reshape(bs, -1, 5, 3).cpu()
            axial_preds.append(preds)
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds_axial[study_id] = preds[b]

    axial_preds = torch.cat(axial_preds, dim=0)
    all_preds = torch.cat((sag_t2_preds, sag_t1_preds, axial_preds), dim=1)
    all_preds_dict = {}
    for i, study_id in enumerate(study_ids):
        all_preds_dict[study_id] = all_preds[i]
    return all_preds_dict


def infer_v24_seed666(  # df_valid,
        data_root,
        study_ids,
        model_dir,
        save_dir,
        device,
        num_workers=8,
        folds=None,
        is_parallel=False,
):
    def _forward_tta(model, x, cond):
        # return model(x, cond)
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x0 = model(x, cond)
        x1 = model(x_flip, cond)
        return (x0 + x1) / 2

    # t1
    model_cfgs = [
        {
            'sub_dir': 'v20_cond_t1/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_666_h128_w128',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': False,
            'data_key': '3_128_128',
            'weight': 0.27957959035255875,
        },
        {
            'sub_dir': 'v20_cond_t1/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_666_h72_w128_with_gru',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_72_128',
            'weight': 0.2299256035844461,
        },
        {
            'sub_dir': 'v20_cond_t1/rexnetr_200.sw_in12k_ft_in1k_z_imgs_3_seed_666_h72_w128_with_gru',
            'backbone': 'rexnetr_200.sw_in12k_ft_in1k',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_72_128',
            'weight': 0.22004214729630475,
        },
        {
            'sub_dir': 'v20_cond_t1/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_666_h72_w128',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': False,
            'data_key': '3_72_128',
            'weight': 0.13829760915067793,
        },
        {
            'sub_dir': 'v20_cond_t1/convnext_tiny.in12k_ft_in1k_384_z_imgs_3_seed_666_h72_w128_with_gru',
            'backbone': 'convnext_tiny.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_72_128',
            'weight': 0.13215504961601252,
        }

    ]
    v24_folds = list(range(5))
    if folds is not None:
        v24_folds = folds

    models = []
    data_keys = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        with_cond = cfg['with_cond']
        with_gru = cfg['with_gru']
        with_level_lstm = cfg['with_level_lstm']
        z_imgs = cfg['z_imgs']
        data_key = cfg['data_key']
        w = cfg['weight']
        for fold in v24_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_ema.pt'
            print('load: ', fn)
            model = build_v20_sag_model(
                backbone,
                with_cond=with_cond,
                with_gru=with_gru,
                with_level_lstm=with_level_lstm,
                z_imgs=z_imgs,
                pretrained=False
            )
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)

            models.append(model)
            weights.append(w)
            data_keys.append(data_key)
    weights_sum = np.sum(weights)
    with open(f'{keypoint_dir}/sag_t1_3d_keypoints_v20.pkl', 'rb') as f:
        pred_sag_keypoints_infos_3d_t1 = pickle.load(f)

    other_crop_size_list = [
        (3, 72, 128),
    ]
    dset = Sag_T1_Dataset_V24(data_root,
                              study_ids,
                              pred_sag_keypoints_infos_3d_t1,
                              test_series_descriptions_fn=test_series_descriptions_fn,
                              image_dir=image_dir,
                              crop_size_h=128,
                              crop_size_w=128,
                              cache_dir=CACHE_DIR,
                              other_crop_size_list=other_crop_size_list
                              )
    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
    )
    all_preds_t1 = {}
    sag_t1_preds = []
    for tensor_d in tqdm.tqdm(dloader):
        with torch.no_grad():
            for k in tensor_d.keys():
                if k not in ['study_id']:
                    tensor_d[k] = tensor_d[k].to(device)
            study_id_list = tensor_d['study_id']
            cond = tensor_d['cond']
            preds = None
            with autocast:
                for i in range(len(models)):
                    p = _forward_tta(models[i], tensor_d[data_keys[i]], cond)
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, n_cond, 5_level, 3
            preds = preds.reshape(bs, -1, 5, 3).cpu()
            sag_t1_preds.append(preds)
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds_t1[study_id] = preds[b]
    sag_t1_preds = torch.cat(sag_t1_preds, dim=0)

    # t2
    model_cfgs = [
        {
            'sub_dir': 'v20_cond_t2/rexnetr_200.sw_in12k_ft_in1k_z_imgs_3_seed_666_h128_w128_level_lstm',
            'backbone': 'rexnetr_200.sw_in12k_ft_in1k',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '3_128_128',
            'weight': 0.2676011522464479,
        },

        {
            'sub_dir': 'v20_cond_t2/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_666_h128_w128_with_gru',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': False,
            'data_key': '3_128_128',
            'weight': 0.25537269026075604,
        },
        {
            'sub_dir': 'v20_cond_t2/pvt_v2_b1.in1k_z_imgs_3_seed_666_h64_w96_with_gru',
            'backbone': 'pvt_v2_b1.in1k',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': True,
            'with_level_lstm': True,
            'data_key': '3_64_96',
            'weight': 0.20531756213739447,
        },
        {
            'sub_dir': 'v20_cond_t2/convnext_tiny.in12k_ft_in1k_z_imgs_5_seed_666_h128_w128_level_lstm',
            'backbone': 'convnext_tiny.in12k_ft_in1k',
            'with_cond': False,
            'z_imgs': 5,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '5_128_128',
            'weight': 0.16238787669016758,
        },
        {
            'sub_dir': 'v20_cond_t2/convnext_tiny.in12k_ft_in1k_z_imgs_3_seed_666_h64_w96_level_lstm',
            'backbone': 'convnext_tiny.in12k_ft_in1k',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '3_64_96',
            'weight': 0.08318722948976684,
        },

        {
            'sub_dir': 'v20_cond_t2/pvt_v2_b2.in1k_z_imgs_3_seed_666_h128_w128_level_lstm',
            'backbone': 'pvt_v2_b2.in1k',
            'with_cond': False,
            'z_imgs': 3,
            'with_gru': False,
            'with_level_lstm': True,
            'data_key': '3_128_128',
            'weight': 0.02613348917546713,
        },
    ]
    v24_folds = list(range(5))
    if folds is not None:
        v24_folds = folds

    models = []
    data_keys = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        with_cond = cfg['with_cond']
        with_gru = cfg['with_gru']
        with_level_lstm = cfg['with_level_lstm']
        z_imgs = cfg['z_imgs']
        data_key = cfg['data_key']
        w = cfg['weight']
        for fold in v24_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_ema.pt'
            print('load: ', fn)
            model = build_v20_sag_model(
                backbone,
                with_cond=with_cond,
                with_gru=with_gru,
                with_level_lstm=with_level_lstm,
                z_imgs=z_imgs,
                pretrained=False
            )
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)

            models.append(model)
            weights.append(w)
            data_keys.append(data_key)
    weights_sum = np.sum(weights)

    with open(f'{keypoint_dir}/sag_t2_3d_keypoints_v20.pkl', 'rb') as f:
        pred_sag_keypoints_infos_3d_t2 = pickle.load(f)

    other_crop_size_list = [
        (3, 64, 96),
        (5, 128, 128),
    ]

    dset = Sag_T2_Dataset_V24(data_root,
                              study_ids,
                              pred_sag_keypoints_infos_3d_t2,
                              test_series_descriptions_fn=test_series_descriptions_fn,
                              image_dir=image_dir,
                              crop_size_h=128,
                              crop_size_w=128,
                              cache_dir=CACHE_DIR,
                              other_crop_size_list=other_crop_size_list
                              )
    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
    )
    all_preds_t2 = {}
    sag_t2_preds = []
    for tensor_d in tqdm.tqdm(dloader):
        with torch.no_grad():
            for k in tensor_d.keys():
                if k not in ['study_id']:
                    tensor_d[k] = tensor_d[k].to(device)
            study_id_list = tensor_d['study_id']
            cond = None  # tensor_d['cond']
            preds = None
            with autocast:
                for i in range(len(models)):
                    # p = _forward_tta(models[i], tensor_d['img'], cond)
                    p = _forward_tta(models[i], tensor_d[data_keys[i]], cond)
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, n_cond, 5_level, 3
            preds = preds.reshape(bs, -1, 5, 3).cpu()
            sag_t2_preds.append(preds)
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds_t2[study_id] = preds[b]
    sag_t2_preds = torch.cat(sag_t2_preds, dim=0)
    # return all_preds_t2

    # axial
    model_cfgs = [
        {
            'sub_dir': 'v24_cond_axial/rexnetr_200.sw_in12k_ft_in1k_z_imgs_5_666',
            'backbone': 'rexnetr_200.sw_in12k_ft_in1k',
            'axial_in_channels': 3,
            'weight': 0.44292661444289017,
            'z_imgs': 5,
            'data_key': 'axial_imgs_5_128_128'
        },
        {
            'sub_dir': 'v24_cond_axial/convnext_tiny.in12k_ft_in1k_384_z_imgs_3_666',
            'backbone': 'convnext_tiny.in12k_ft_in1k',
            'axial_in_channels': 1,
            'weight': 0.29758735999956626,
            'z_imgs': 3,
            'data_key': 'axial_imgs_3_128_128'
        },

        {
            'sub_dir': 'v24_cond_axial/convnext_small.in12k_ft_in1k_384_z_imgs_3_666',
            'backbone': 'convnext_small.in12k_ft_in1k_384',
            'axial_in_channels': 1,
            'weight': 0.2384201128638661,
            'z_imgs': 3,
            'data_key': 'axial_imgs_3_128_128'
        },
        {
            'sub_dir': 'v24_cond_axial/densenet161_z_imgs_3_666',
            'backbone': 'densenet161',
            'axial_in_channels': 1,
            'weight': 0.021065912693677462,
            'z_imgs': 3,
            'data_key': 'axial_imgs_3_128_128'
        },

    ]
    v24_folds = list(range(5))
    if folds is not None:
        v24_folds = folds

    models = []
    data_keys = []
    weights = []
    for cfg in model_cfgs:
        sub_dir = cfg['sub_dir']
        backbone = cfg['backbone']
        data_key = cfg['data_key']
        axial_in_channels = cfg['axial_in_channels']
        w = cfg['weight']
        for fold in v24_folds:
            fn = os.path.join(model_dir, sub_dir)
            fn = fn + f'/best_wll_model_fold-{fold}_ema.pt'
            print('load: ', fn)
            model = Axial_HybridModel_24(backbone,
                                         backbone,
                                         pretrained=False,
                                         axial_in_channels=axial_in_channels
                                         )
            model.load_state_dict(torch.load(fn))
            model.to(device)
            model.eval()
            if is_parallel:
                model = nn.DataParallel(model)

            models.append(model)
            weights.append(w)
            data_keys.append(data_key)
    weights_sum = np.sum(weights)

    with open(f'{keypoint_dir}/axial_3d_keypoints_v24.pkl', 'rb') as f:
        axial_3d_keypoints_v24 = pickle.load(f)

    other_crop_size_list = [
        (5, 128, 128),
    ]

    dset = Axial_Cond_Dataset_Multi_V24(data_root,
                                        study_ids,
                                        axial_3d_keypoints_v24,
                                        pred_sag_keypoints_infos_3d_t2,
                                        test_series_descriptions_fn=test_series_descriptions_fn,
                                        image_dir=image_dir,
                                        z_imgs=3,
                                        other_z_imgs_list=other_crop_size_list,
                                        cache_dir=CACHE_DIR,
                                        )

    dloader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=num_workers,
    )
    all_preds_axial = {}
    axial_preds = []
    for tensor_d in tqdm.tqdm(dloader):
        with torch.no_grad():
            for k in tensor_d.keys():
                if k not in ['study_id']:
                    tensor_d[k] = tensor_d[k].to(device)
            study_id_list = tensor_d['study_id']
            preds = None
            with autocast:
                for i in range(len(models)):
                    p = models[i](tensor_d['sag_t2'], tensor_d[data_keys[i]])
                    if preds is None:
                        preds = (p.float() * weights[i])
                    else:
                        preds += (p.float() * weights[i])
            preds = preds / weights_sum
            bs, _ = preds.shape
            # bs, n_cond, 5_level, 3
            preds = preds.reshape(bs, -1, 5, 3).cpu()
            axial_preds.append(preds)
            for b in range(bs):
                study_id = int(study_id_list[b])
                all_preds_axial[study_id] = preds[b]

    axial_preds = torch.cat(axial_preds, dim=0)
    all_preds = torch.cat((sag_t2_preds, sag_t1_preds, axial_preds), dim=1)
    all_preds_dict = {}
    for i, study_id in enumerate(study_ids):
        all_preds_dict[study_id] = all_preds[i]
    return all_preds_dict


def get_df_valid(data_root, fold=0, with_patriot=False):
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    if with_patriot:
        folds_t2s_each = pd.read_csv("../patriot/folds_t2s_each.csv")
        valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index
        folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)
        df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)
        return df_valid
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=666)
        for i, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
            if i == fold:
                df_valid = df.iloc[val_idx]
                return df_valid.copy().copy().reset_index(drop=True)


def infer_all(data_root,
              model_dir,
              model_dir2,
              save_dir,
              is_local_val=True,
              local_val_fold=0,
              ):
    df = pd.read_csv(f'{data_root}/{test_series_descriptions_fn}')
    study_ids = df['study_id'].unique().tolist()
    # study_ids = study_ids[:8]

    if is_local_val:
        fold = local_val_fold
        df_valid = get_df_valid(data_root, fold, with_patriot=True)
        study_ids = list(df_valid['study_id'].unique())
        labels = df_valid.iloc[:, 1:].values
        labels = torch.from_numpy(labels)

    device_0 = torch.device('cuda:0')

    is_parallel = torch.cuda.device_count() > 1 and len(study_ids) >= 2
    print('is_parallel: ', is_parallel)
    folds = None
    if is_local_val:
        folds = [local_val_fold]

    pred_dict_v2 = infer_v2(data_root, study_ids, model_dir,
                            save_dir, device_0, N_WORKERS, folds, is_parallel)

    pred_dict_v24 = infer_v24(data_root, study_ids, model_dir,
                              save_dir, device_0, N_WORKERS, folds, is_parallel)

    print(len(study_ids))
    fold_preds = []
    for study_id in study_ids:
        pred = pred_dict_v2[study_id]
        fold_preds.append(pred.unsqueeze(0))
    fold_preds = torch.cat(fold_preds, dim=0)

    print('fold_preds shape: ', fold_preds.shape)

    fold_preds_v24 = []
    for study_id in study_ids:
        pred = pred_dict_v24[study_id]
        fold_preds_v24.append(pred.unsqueeze(0))
    fold_preds_v24 = torch.cat(fold_preds_v24, dim=0)

    fold_preds_en = fold_preds
    fold_preds_en[:, 0:1] = 0.5896 * fold_preds[:, 0:1] + 0.4104 * fold_preds_v24[:, 0:1]
    fold_preds_en[:, 1:3] = 0.09 * fold_preds[:, 1:3] + 0.91 * fold_preds_v24[:, 1:3]
    fold_preds_en[:, 3:5] = 0.38 * fold_preds[:, 3:5] + 0.62 * fold_preds_v24[:, 3:5]

    if is_local_val:
        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        t2_labels = labels[:, :5].reshape(-1, 1, 5).reshape(-1)
        t2_pred = fold_preds_en[:, 0:1, :, :].reshape(-1, 3)
        t2_ce = criterion2(t2_pred, t2_labels)

        t1_labels = labels[:, 5:15].reshape(-1, 2, 5).reshape(-1)
        t1_pred = fold_preds_en[:, 1:3, :, :].reshape(-1, 3)
        t1_ce = criterion2(t1_pred, t1_labels)

        axial_labels = labels[:, 15:].reshape(-1, 2, 5).reshape(-1)
        axial_pred = fold_preds_en[:, 3:, :, :].reshape(-1, 3)
        axial_ce = criterion2(axial_pred, axial_labels)

        fold_preds_en = nn.Softmax(dim=-1)(fold_preds_en).cpu().numpy()

        l5 = fold_preds_en[:, :, 4, :]
        l4 = fold_preds_en[:, :, 3, :]
        l3 = fold_preds_en[:, :, 2, :]
        l2 = fold_preds_en[:, :, 1, :]
        l1 = fold_preds_en[:, :, 0, :]

        c, val_df_ = make_calc(study_ids,
                               l5, l4, l3, l2, l1, df_valid)
        print(f"metric  {c}")
        print(f"t2_ce:  {t2_ce}, t1_ce: {t1_ce},axial_ce: {axial_ce} ")
        return c, t2_ce, t1_ce, axial_ce

    fold_preds_en = nn.Softmax(dim=-1)(fold_preds_en).cpu().numpy()
    return fold_preds_en


def infer_all_666(data_root,
                  model_dir,
                  model_dir2,
                  save_dir,
                  is_local_val=True,
                  local_val_fold=0,
                  ):
    df = pd.read_csv(f'{data_root}/{test_series_descriptions_fn}')
    study_ids = df['study_id'].unique().tolist()
    # study_ids = study_ids[:8]

    if is_local_val:
        fold = local_val_fold
        df_valid = get_df_valid(data_root, fold, with_patriot=False)
        study_ids = list(df_valid['study_id'].unique())
        labels = df_valid.iloc[:, 1:].values
        labels = torch.from_numpy(labels)

    device_0 = torch.device('cuda:0')

    is_parallel = torch.cuda.device_count() > 1 and len(study_ids) >= 2
    print('is_parallel: ', is_parallel)
    folds = None
    if is_local_val:
        folds = [local_val_fold]

    pred_dict_v2 = infer_v2_seed666(data_root, study_ids, model_dir2,
                                    save_dir, device_0, N_WORKERS, folds, is_parallel)

    pred_dict_v24 = infer_v24_seed666(data_root, study_ids, model_dir2,
                                      save_dir, device_0, N_WORKERS, folds, is_parallel)

    print(len(study_ids))
    fold_preds = []
    for study_id in study_ids:
        pred = pred_dict_v2[study_id]
        fold_preds.append(pred.unsqueeze(0))
    fold_preds = torch.cat(fold_preds, dim=0)

    fold_preds_v24 = []
    for study_id in study_ids:
        pred = pred_dict_v24[study_id]
        fold_preds_v24.append(pred.unsqueeze(0))
    fold_preds_v24 = torch.cat(fold_preds_v24, dim=0)

    fold_preds_en = fold_preds
    fold_preds_en[:, 0:1] = 0.55 * fold_preds[:, 0:1] + 0.45 * fold_preds_v24[:, 0:1]
    fold_preds_en[:, 1:3] = 0.1 * fold_preds[:, 1:3] + 0.9 * fold_preds_v24[:, 1:3]
    fold_preds_en[:, 3:5] = 0.38 * fold_preds[:, 3:5] + 0.62 * fold_preds_v24[:, 3:5]

    if is_local_val:
        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        t2_labels = labels[:, :5].reshape(-1, 1, 5).reshape(-1)
        t2_pred = fold_preds_en[:, 0:1, :, :].reshape(-1, 3)
        t2_ce = criterion2(t2_pred, t2_labels)

        t1_labels = labels[:, 5:15].reshape(-1, 2, 5).reshape(-1)
        t1_pred = fold_preds_en[:, 1:3, :, :].reshape(-1, 3)
        t1_ce = criterion2(t1_pred, t1_labels)

        axial_labels = labels[:, 15:].reshape(-1, 2, 5).reshape(-1)
        axial_pred = fold_preds_en[:, 3:, :, :].reshape(-1, 3)
        axial_ce = criterion2(axial_pred, axial_labels)

        fold_preds_en = nn.Softmax(dim=-1)(fold_preds_en).cpu().numpy()

        l5 = fold_preds_en[:, :, 4, :]
        l4 = fold_preds_en[:, :, 3, :]
        l3 = fold_preds_en[:, :, 2, :]
        l2 = fold_preds_en[:, :, 1, :]
        l1 = fold_preds_en[:, :, 0, :]

        c, val_df_ = make_calc(study_ids,
                               l5, l4, l3, l2, l1, df_valid)
        print(f"metric  {c}")
        print(f"t2_ce:  {t2_ce}, t1_ce: {t1_ce},axial_ce: {axial_ce} ")
        return c, t2_ce, t1_ce, axial_ce

    fold_preds_en = nn.Softmax(dim=-1)(fold_preds_en).cpu().numpy()
    return fold_preds_en


if __name__ == '__main__':
    args = get_args()
    data_root = args.data_root
    model_dir = args.model_dir
    model_dir2 = args.model_dir2
    save_dir = args.save_dir
    if IS_KAGGLE:
        preds = infer_all(data_root, model_dir, model_dir2, save_dir, is_local_val=False)
        preds = preds.reshape(-1, 3)
        print('preds shape: ', preds.shape)
        print(preds)
        exit(0)

    scores = []
    t2_ce_list = []
    t1_ce_list = []
    axial_ce_list = []

    all_fold_preds = []
    all_labels = []
    all_fold_preds_v24 = []
    for fold in range(0, 5):
        # t2_ce = infer_all(data_root, model_dir, save_dir, is_local_val=True,
        #                                       local_val_fold=fold)
        # t2_ce_list.append(t2_ce)
        c, t2_ce, t1_ce, axial_ce = infer_all_666(data_root, model_dir, model_dir2,
                                                  save_dir,
                                                  is_local_val=True,
                                                  local_val_fold=fold)
        scores.append(c)
        t2_ce_list.append(t2_ce)
        t1_ce_list.append(t1_ce)
        axial_ce_list.append(axial_ce)
        # all_labels.append(labels)
        # all_fold_preds.append(fold_preds)
        # all_fold_preds_v24.append(fold_preds_v24)

    print('score: ', scores)
    print('mean comp metric: ', np.mean(scores))

    # print('t2_ce_list: ', t2_ce_list)
    print('mean t2 ce: ', np.mean(t2_ce_list))
    print('mean t1 ce: ', np.mean(t1_ce_list))
    print('mean axial ce: ', np.mean(axial_ce_list))

    # all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    # all_fold_preds = torch.cat(all_fold_preds, dim=0).cpu().numpy()
    # all_fold_preds_v24 = torch.cat(all_fold_preds_v24, dim=0).cpu().numpy()
    # np.save('./all_labels.npy', all_labels)
    # np.save('./all_fold_preds.npy', all_fold_preds)
    # np.save('./all_fold_preds_v24.npy', all_fold_preds_v24)
