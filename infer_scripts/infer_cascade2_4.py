# -*- coding: utf-8 -*-
import os
import pickle

import tqdm
from sklearn.model_selection import KFold

os.environ['OMP_NUM_THREADS'] = '1'
from infer_scripts.v24.infer_utils import *
from src.utils.comm import setup_seed
import warnings
import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


###
from patriot.metric import CALC_score
from train_scripts.v20.hard_list import  all_hard_list

def get_level_target(valid_df, level=0):
    names = ["spinal_canal_stenosis",
             "left_neural_foraminal_narrowing",
             "right_neural_foraminal_narrowing",
             "left_subarticular_stenosis",
             "right_subarticular_stenosis"
             ]
    _idx_to_level_name = {
        1: 'l1_l2',
        2: 'l2_l3',
        3: 'l3_l4',
        4: 'l4_l5',
        5: 'l5_s1',
    }
    for i in range(len(names)):
        names[i] = '{}_{}'.format(names[i], _idx_to_level_name[level])
    return valid_df[names].values


def make_calc(test_stusy, l5, l4, l3, l2, l1, folds_tmp):
    new_df = pd.DataFrame()
    data_root = '/home/hw/ssd_01_new/kaggle/rsna-2024-lumbar-spine-degenerative-classification/'
    tra_df = list(pd.read_csv(f"{data_root}/train.csv").columns[1:])
    col = []
    c_ = []
    level = []
    for i in test_stusy:
        for j in tra_df:
            col.append(f"{i}_{j}")
            c_.append(f"{i}")
            level.append(j.split("_")[-2])

    # print(level[:10])

    new_df["row_id"] = col
    new_df["study_id"] = c_
    new_df["level"] = level

    new_df["level"] = new_df["level"].astype("str")
    new_df["row_id"] = new_df["row_id"].astype("str")
    new_df["normal_mild"] = 0
    new_df["moderate"] = 0
    new_df["severe"] = 0
    new_df___ = []
    name__2 = {"l5": 0, "l4": 1, "l3": 2, "l2": 3, "l1": 4}
    for pred, level in zip([l5, l4, l3, l2, l1], [5, 4, 3, 2, 1]):
        name_ = f'l{level}'
        new_df_ = new_df[new_df["level"] == name_]
        # fold_tmp_ = folds_tmp[folds_tmp["level"] == name__2[name_]][
        #     ["spinal_canal_stenosis", "left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing",
        #      "left_subarticular_stenosis", "right_subarticular_stenosis"]].values

        fold_tmp_ = get_level_target(folds_tmp, level)
        new_df_[["normal_mild", "moderate", "severe"]] = pred.reshape(-1, 3)
        new_df_["GT"] = fold_tmp_.reshape(-1, )
        new_df___.append(new_df_)

    new_df = pd.concat(new_df___).sort_values("row_id").reset_index(drop=True)

    new_df = new_df[new_df["GT"] != -100].reset_index(drop=True)
    # この時点でauc計算でも良い？
    GT = new_df.iloc[:, [0, -1]].copy()
    GT[["normal_mild", "moderate", "severe"]] = np.eye(3)[GT["GT"].to_numpy().astype(np.uint8)]
    GT["sample_weight"] = 2 ** GT["GT"].to_numpy()

    GT = GT.iloc[:, [0, 2, 3, 4, 5]]
    metirc_ = CALC_score(GT, new_df.iloc[:, [0, 3, 4, 5]], row_id_column_name="row_id")
    return metirc_, new_df


##

if __name__ == '__main__':

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)

    hard_axial_study_id_list = [3008676218,
                                391103067,
                                953639220,
                                2460381798,
                                2690161683,
                                3650821463,
                                3949892272,
                                677672203,  # 左右点标注反了
                                ]
    df = df[~df['study_id'].isin(hard_axial_study_id_list)]
    df = df[~df['study_id'].isin(all_hard_list)]

    SEED = 8620
    setup_seed(SEED, deterministic=True)
    skf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    #### val
    cv = 0

    y_preds_ema = []
    labels = []
    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion2 = nn.CrossEntropyLoss(weight=weights)
    scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        # if fold == 0:
        #    continue
        print('#' * 30)
        print(f'start fold{fold}')
        print('#' * 30)
        # df_valid = df.iloc[val_idx]

        folds_t2s_each = pd.read_csv("../rsna2024lsdc-patriot-modified/folds_t2s_each.csv")
        valid_idx = folds_t2s_each[folds_t2s_each['fold'] == fold].index
        folds_t2s_each_valid = folds_t2s_each.loc[valid_idx].copy().reset_index(drop=True)
        df_valid = df[df['study_id'].isin(folds_t2s_each_valid['study_id'])].copy().reset_index(drop=True)
        #df_valid = df_valid[:8]
        '''
        # infer the sag_t2 2d keypoint
        print('infer t2 3d keypoints ..')
        study_ids = list(df_valid['study_id'].unique())
        pred_sag_keypoints_infos_3d_t2 = {}
        models = load_3d_keypoints_models(
            model_dir='wkdir/v20/sag_t2_3d_keypoints/densenet161_lr_0.0006_sag_T2/',
            num_classes=15
        )
        dset = Sag_3D_Point_Dataset_V24(data_root,
                 study_ids,
                 series_description='Sagittal T2/STIR', )
        dloader = DataLoader(dset, batch_size=8, num_workers=8)
        with tqdm.tqdm(dloader, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['imgs'].cuda()
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
                        if study_id not in pred_sag_keypoints_infos_3d_t2:
                            pred_sag_keypoints_infos_3d_t2[study_id] = {}
                        pred_sag_keypoints_infos_3d_t2[study_id][series_id] = pts
        ##
        print('infer t1 3d keypoints ..')
        pred_sag_keypoints_infos_3d_t1 = {}
        models = load_3d_keypoints_models(
            model_dir='wkdir/v20/sag_t2_3d_keypoints/densenet161_lr_0.0006_sag_T1/',
            num_classes=30
        )
        dset = Sag_3D_Point_Dataset_V24(data_root,
                                        study_ids,
                                        series_description='Sagittal T1', )
        dloader = DataLoader(dset, batch_size=8, num_workers=8)
        with tqdm.tqdm(dloader, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_dict in enumerate(pbar):
                    x = tensor_dict['imgs'].cuda()
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
                        if study_id not in pred_sag_keypoints_infos_3d_t1:
                            pred_sag_keypoints_infos_3d_t1[study_id] = {}
                        pred_sag_keypoints_infos_3d_t1[study_id][series_id] = pts
        '''
        pred_sag_keypoints_infos_3d_t1 = pickle.load(
            open(f'{data_root}/v20_sag_T1_3d_keypoints_en5_densenet161.pkl', 'rb'))
        pred_sag_keypoints_infos_3d_t2 = pickle.load(
            open(f'{data_root}/v20_sag_T2_3d_keypoints_en5_densenet161.pkl', 'rb'))

        print('infer sag_t2 cond ...')
        study_ids = list(df_valid['study_id'].unique())
        dset = Sag_T2_Dataset_V24(data_root,
                                  study_ids,
                                  pred_sag_keypoints_infos_3d_t2,
                                  crop_size_h=72,
                                  crop_size_w=128
                                  )
        dloader = DataLoader(
            dset,
            batch_size=8,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=12,
        )

        model = build_sag_model("convnext_small.in12k_ft_in1k_384",
                                     pretrained=False).cuda()
        fname = f'./wkdir/v24/sag_t2/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h72_w128/best_wll_model_fold-{fold}_ema.pt'
        model.load_state_dict(torch.load(fname))
        model.eval()

        sag_t2_preds = []
        with tqdm.tqdm(dloader, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_d in enumerate(pbar):
                    sag_t2 = tensor_d['s_t2'].cuda()
                    with autocast:
                        ye = model(sag_t2)
                        bs, _ = ye.shape
                        # bs, 1_cond, 5_level, 3
                        ye = ye.reshape(bs, 1, 5, 3)
                        sag_t2_preds.append(ye)
        sag_t2_preds = torch.cat(sag_t2_preds, dim=0)

        #
        print('infer sag_t1 cond ...')
        study_ids = list(df_valid['study_id'].unique())
        dset = Sag_T1_Dataset_V24(data_root,
                                  study_ids,
                                  pred_sag_keypoints_infos_3d_t1,
                                  crop_size_h=128,
                                  crop_size_w=128
                                  )
        dloader = DataLoader(
            dset,
            batch_size=8,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=12,
        )

        model = build_sag_model("convnext_small.in12k_ft_in1k_384",
                                with_level_lstm=False,
                                pretrained=False).cuda()
        fname = f'./wkdir_final/v24_cond_t1//convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128/best_wll_model_fold-{fold}_ema.pt'
        model.load_state_dict(torch.load(fname))
        model.eval()

        sag_t1_preds = []
        with tqdm.tqdm(dloader, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_d in enumerate(pbar):
                    sag_t1 = tensor_d['s_t1'].cuda()
                    with autocast:
                        ye = model(sag_t1)
                        bs, _ = ye.shape
                        # bs, 2_cond, 5_level, 3
                        ye = ye.reshape(bs, 2, 5, 3)
                        sag_t1_preds.append(ye)
        sag_t1_preds = torch.cat(sag_t1_preds, dim=0)

        study_ids = list(df_valid['study_id'].unique())

        '''
        ## infer the axial 3d keypoint
        print('infer the axial 3d keypoint ..')
        dset = Axial_Level_Dataset_Multi_V24(data_root, study_ids, )
        dloader = DataLoader(dset, batch_size=1, num_workers=8,
                             collate_fn=axial_v24_collate_fn)

        models = load_axial_v24_level_models()
        xy_models = load_axial_v24_xy_models()

        axial_pred_keypoints_info = {}
        for data_dict in tqdm.tqdm(dloader):
            study_id, series_id_list, pred_keypoints = axial_v24_infer_z(models, data_dict)
            pred_keypoints = axial_v24_infer_xy(xy_models, pred_keypoints, data_dict)
            study_id = int(study_id)
            axial_pred_keypoints_info[study_id] = {}
            for i, sid in enumerate(series_id_list):
                sid = int(sid)
                axial_pred_keypoints_info[study_id][sid] = {
                    'points': pred_keypoints[i]  # 2, 5, 4
                }
        '''
        axial_pred_keypoints_info = pickle.load(
            open(f'{data_root}/v24_axial_pred_en5_no_ipp2.pkl', 'rb'))
        #
        print('infer axial cond ...')
        study_ids = list(df_valid['study_id'].unique())
        dset = Axial_Cond_Dataset_Multi_V24(data_root,
                                            study_ids,
                                            axial_pred_keypoints_info,
                                            pred_sag_keypoints_infos_3d_t2, )
        dloader = DataLoader(
            dset,
            batch_size=8,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=12,

        )

        model = Axial_HybridModel_24("convnext_small.in12k_ft_in1k_384",
                                     "convnext_small.in12k_ft_in1k_384",
                                     pretrained=False).cuda()
        fname = f'wkdir/v24_single/axial/convnext_small.in12k_ft_in1k_384_z_imgs_5/best_wll_model_fold-{fold}_ema.pt'
        model.load_state_dict(torch.load(fname))
        model.eval()

        axial_preds = []
        tmp_study_ids = []
        with tqdm.tqdm(dloader, leave=True) as pbar:
            with torch.no_grad():
                for idx, tensor_d in enumerate(pbar):
                    sag_t2 = tensor_d['sag_t2'].cuda()
                    axial_imgs = tensor_d['axial_imgs'].cuda()
                    study_id_list = tensor_d['study_id']
                    for study_id in study_id_list:
                        tmp_study_ids.append(int(study_id))
                    with autocast:
                        ye = model(sag_t2, axial_imgs)
                        bs, _ = ye.shape
                        # bs, 2_cond, 5_level, 3
                        ye = ye.reshape(bs, 2, 5, 3)
                        axial_preds.append(ye)
        axial_preds = torch.cat(axial_preds, dim=0)

        # bs, 5_cond, 5_level, 3
        fold_preds = torch.cat((sag_t2_preds, sag_t1_preds, axial_preds), dim=1)
        # print('fold_preds shape: ', fold_preds.shape)
        # print(fold_preds[0][3])
        # print('study_id: ', tmp_study_ids[0])
        # exit(0)

        fold_preds = nn.Softmax(dim=-1)(fold_preds).cpu().numpy()
        l5 = fold_preds[:, :, 4, :]
        l4 = fold_preds[:, :, 3, :]
        l3 = fold_preds[:, :, 2, :]
        l2 = fold_preds[:, :, 1, :]
        l1 = fold_preds[:, :, 0, :]

        c, val_df_ = make_calc(df_valid["study_id"].unique(),
                               l5, l4, l3, l2, l1, df_valid)
        print(f"metric  {c}")
        scores.append(c)

    print(scores)
    print('mean: ', np.mean(scores))
