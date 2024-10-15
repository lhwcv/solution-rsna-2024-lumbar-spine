# -*- coding: utf-8 -*-
import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn as nn

from train_scripts.data_path import DATA_ROOT
from train_scripts.v24.axial_model import Axial_Level_Cls_Model
from  train_scripts.v24.models import RSNA24Model_Keypoint_2D
from train_scripts.v24.axial_data import Axial_Level_Dataset_Multi, data_to_cuda, collate_fn
import albumentations as A

autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)


def infer(models, data_dict):
    data_dict = data_to_cuda(data_dict)
    xy_pred = None
    sparse_pred = None
    coords = data_dict[0]['coords'].cpu()
    index_info = data_dict[0]['index_info']
    study_id = data_dict[0]['study_id']
    with torch.no_grad():
        with autocast:
            for i in range(len(models)):
                p = models[i](data_dict)
                if xy_pred is None:
                    xy_pred = p['xy_pred']
                    sparse_pred = p['sparse_pred']
                else:
                    xy_pred += p['xy_pred']
                    sparse_pred += p['sparse_pred']
            xy_pred /= len(models)
            sparse_pred /= len(models)
    #
    _, h, w = data_dict[0]['arr'].shape
    z_prob = sparse_pred[0].sigmoid().cpu()
    xy_pred = xy_pred[0].cpu()
    #print(xy_pred.shape)
    # n_seg, 2, 5, 3
    n_seg = len(index_info)
    pred_keypoints = np.zeros((n_seg, 2, 5, 4))
    for n in range(n_seg):
        z0, z1 = index_info[n]
        z_len = z1 - z0
        zs = coords[z0:z1]
        prob = z_prob[z0:z1]
        #print('prob shape: ', prob.shape)
        xy = xy_pred[z0:z1]
        for i in range(5):
            left_conf = prob[:, 0, i].max()
            right_conf = prob[:, 1, i].max()
            recover_left_z = (prob[:, 0, i] * zs).sum() / prob[:, 0, i].sum()
            recover_right_z = (prob[:, 1, i] * zs).sum() / prob[:, 1, i].sum()

            if recover_left_z < 0:
                recover_left_z = 0
            if recover_left_z > z_len-1:
                recover_left_z = z_len - 1

            if recover_right_z < 0:
                recover_right_z = 0
            if recover_right_z > z_len-1:
                recover_right_z = z_len - 1

            left_xy = xy[:, 0, :][int(np.round(recover_left_z))]
            right_xy = xy[:, 1, :][int(np.round(recover_right_z))]

            #print('left_xy: ', left_xy)

            pred_keypoints[n, 0, i, 0] = left_xy[0] / w
            pred_keypoints[n, 0, i, 1] = left_xy[1] / h
            pred_keypoints[n, 0, i, 2] = recover_left_z / z_len
            pred_keypoints[n, 0, i, 3] = left_conf

            pred_keypoints[n, 1, i, 0] = right_xy[0] / w
            pred_keypoints[n, 1, i, 1] = right_xy[1] / h
            pred_keypoints[n, 1, i, 2] = recover_right_z / z_len
            pred_keypoints[n, 1, i, 3] = right_conf
    series_id_list = data_dict[0]['series_id_list']
    return study_id, series_id_list, pred_keypoints


def infer_z(models, data_dict):
    data_dict = data_to_cuda(data_dict)

    sparse_pred = None
    coords = data_dict[0]['coords'].cpu()
    index_info = data_dict[0]['index_info']
    study_id = data_dict[0]['study_id']
    with torch.no_grad():
        with autocast:
            for i in range(len(models)):
                p = models[i](data_dict)
                if sparse_pred is None:
                    sparse_pred = p['sparse_pred']
                else:
                    sparse_pred += p['sparse_pred']

            sparse_pred /= len(models)
    #
    _, h, w = data_dict[0]['arr'].shape
    z_prob = sparse_pred[0].sigmoid().cpu()

    #print(xy_pred.shape)
    # n_seg, 2, 5, 3
    n_seg = len(index_info)
    pred_keypoints = np.zeros((n_seg, 2, 5, 2))
    for n in range(n_seg):
        z0, z1 = index_info[n]
        z_len = z1 - z0
        zs = coords[z0:z1]
        prob = z_prob[z0:z1]

        for i in range(5):
            left_conf = prob[:, 0, i].max()
            right_conf = prob[:, 1, i].max()
            recover_left_z = (prob[:, 0, i] * zs).sum() / prob[:, 0, i].sum()
            recover_right_z = (prob[:, 1, i] * zs).sum() / prob[:, 1, i].sum()

            if recover_left_z < 0:
                recover_left_z = 0
            if recover_left_z > z_len-1:
                recover_left_z = z_len - 1

            if recover_right_z < 0:
                recover_right_z = 0
            if recover_right_z > z_len-1:
                recover_right_z = z_len - 1

            pred_keypoints[n, 0, i, 0] = recover_left_z / z_len
            pred_keypoints[n, 0, i, 1] = left_conf
            pred_keypoints[n, 1, i, 0] = recover_right_z / z_len
            pred_keypoints[n, 1, i, 1] = right_conf

    series_id_list = data_dict[0]['series_id_list']
    return study_id, series_id_list, pred_keypoints

# def load_models():
#     model_dir = './wkdir/v24/pretrain_axial_level_cls_no_ipp2/convnext_small.in12k_ft_in1k_384/'
#     fns = [
#         model_dir + '/best_fold_0_ema.pt',
#         # model_dir + '/best_fold_1_ema.pt',
#         # model_dir + '/best_fold_2_ema.pt',
#         # model_dir + '/best_fold_3_ema.pt',
#         # model_dir + '/best_fold_4_ema.pt',
#     ]
#     models = []
#     for fn in fns:
#         print('load: ', fn)
#         model = Axial_Level_Cls_Model('convnext_small.in12k_ft_in1k_384',
#                                       pretrained=False).cuda()
#         model.load_state_dict(torch.load(fn))
#         model.eval()
#         models.append(model)
#     return models

# def load_xy_models():
#     model_dir = './wkdir/v24/axial_2d_keypoints/densenet161_lr_0.0006/'
#     fns = [
#         model_dir + '/best_fold_0_ema.pt',
#         model_dir + '/best_fold_1_ema.pt',
#         model_dir + '/best_fold_2_ema.pt',
#         model_dir + '/best_fold_3_ema.pt',
#         model_dir + '/best_fold_4_ema.pt',
#     ]
#     models = []
#     for fn in fns:
#         print('load: ', fn)
#         model = RSNA24Model_Keypoint_2D('densenet161',
#                                          pretrained=False,
#                                          num_classes=4).cuda()
#         model.load_state_dict(torch.load(fn))
#         model.eval()
#         models.append(model)
#     return models

def load_models():
    model_dir = './wkdir_final/keypoint_3d_v24_axial//level_cls/convnext_small.in12k_ft_in1k_384/'
    fns = [
        model_dir + '/best_fold_0_ema.pt',
        #model_dir + '/best_fold_1_ema.pt',
        # model_dir + '/best_fold_2_ema.pt',
        # model_dir + '/best_fold_3_ema.pt',
        # model_dir + '/best_fold_4_ema.pt',
    ]
    models = []
    for fn in fns:
        print('load: ', fn)
        model = Axial_Level_Cls_Model('convnext_small.in12k_ft_in1k_384',
                                      pretrained=False).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models.append(model)
    return models

def load_xy_models():
    model_dir = './wkdir_final/keypoint_3d_v24_axial/axial_2d_keypoints/densenet161_lr_0.0006'
    fns = [
        model_dir + '/best_fold_0_ema.pt',
        model_dir + '/best_fold_1_ema.pt',
        model_dir + '/best_fold_2_ema.pt',
        model_dir + '/best_fold_3_ema.pt',
        model_dir + '/best_fold_4_ema.pt',
    ]
    models = []
    for fn in fns:
        print('load: ', fn)
        model = RSNA24Model_Keypoint_2D('densenet161',
                                         pretrained=False,
                                         num_classes=4).cuda()
        model.load_state_dict(torch.load(fn))
        model.eval()
        models.append(model)
    return models


class LossFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.xy_dis_loss_fn = nn.SmoothL1Loss()

    def forward(self, pred_dict):
        z_label = pred_dict['z_label']
        level_cls_pred = pred_dict['level_cls_pred']
        bs, z_len, _ = level_cls_pred.shape
        z_label = z_label.reshape(-1)
        level_cls_pred = level_cls_pred.reshape(bs * z_len, -1)
        level_cls_loss = self.ce(level_cls_pred, z_label)

        #dense_xy_keypoints = pred_dict['dense_xy_keypoints'] / 128.0
        #xy_pred = pred_dict['xy_pred'] / 128.0
        #dense_xy_mask = pred_dict['dense_xy_mask']

        # print(dense_xy_keypoints.shape)
        # print(xy_pred.shape)
        # print(dense_xy_mask.shape)

        #dense_xy_keypoints = dense_xy_keypoints * dense_xy_mask
        #xy_pred = xy_pred * dense_xy_mask
        #reg_xy_loss = self.xy_dis_loss_fn(xy_pred, dense_xy_keypoints)

        sparse_pred = pred_dict['sparse_pred']
        sparse_label = pred_dict['sparse_label']
        dense_z_mask = pred_dict['dense_z_mask']
        bce_loss = self.bce(sparse_pred * dense_z_mask, sparse_label * dense_z_mask)

        total_loss = level_cls_loss + bce_loss# + reg_xy_loss

        # decode z
        xyz_keypoints = pred_dict['xyz_keypoints']
        index_info = pred_dict['index_info']
        coords = pred_dict['coords']

        z_prob = sparse_pred[0].sigmoid()
        z_err = 0
        z_err_n = 0
        # n_seg, 2, 5, 3
        n_seg, _, _, _ = xyz_keypoints.shape

        for n in range(n_seg):
            z0, z1 = index_info[n]
            zs = coords[z0:z1]
            prob = z_prob[z0:z1]
            for i in range(5):
                gt_left_z = xyz_keypoints[n, 0, i, 2]
                gt_right_z = xyz_keypoints[n, 1, i, 2]
                recover_left_z = (prob[:, 0, i] * zs).sum() / prob[:, 0, i].sum()
                recover_right_z = (prob[:, 1, i] * zs).sum() / prob[:, 1, i].sum()
                if gt_left_z > 0:
                    err = torch.abs(gt_left_z - recover_left_z)
                    z_err += err
                    z_err_n += 1
                    # print('err: ', err.item())
                if gt_right_z > 0:
                    err = torch.abs(gt_right_z - recover_right_z)
                    z_err += err
                    z_err_n += 1
                    # print('err: ', err.item())
        z_err = z_err / (z_err_n + 1)

        total_loss += 0.1 * z_err

        return {
            'total_loss': total_loss,
            'level_cls_loss': level_cls_loss,
            'z_err': z_err,
            #'reg_xy_loss': reg_xy_loss
        }


def val(model, dataloader):
    total_loss = 0
    total_xy_loss = 0
    total_z_loss = 0
    y_preds = []
    labels = []
    loss_criterion = LossFunc()
    model.eval()
    with tqdm.tqdm(dataloader, leave=True) as pbar:
        with torch.no_grad():
            for idx, tensor_dict in enumerate(pbar):
                tensor_dict = data_to_cuda(tensor_dict)
                with autocast:
                    pred_dict = model(tensor_dict)
                    loss_dict = loss_criterion(pred_dict)
                    loss = loss_dict['total_loss']

                    total_loss += loss.item()
                    total_z_loss += loss_dict['z_err'].item()
                    #total_xy_loss += loss_dict['reg_xy_loss'].item()

                    pred = pred_dict['level_cls_pred'].reshape(-1, 5).cpu()
                    label = pred_dict['z_label'].reshape(-1).cpu()

                    y_preds.append(pred)
                    labels.append(label)

                    pred = pred.float().softmax(dim=-1).numpy()  # n, 5
                    label = label.numpy()
                    mask = np.where(label != -100)
                    pred = pred[mask]
                    label = label[mask]
                    pred_cls = np.argmax(pred, axis=1)
                    acc = accuracy_score(label, pred_cls)
                    pred_prob = []
                    for i in range(len(pred_cls)):
                        pred_prob.append(pred[i, pred_cls[i]])
                    # print('acc: ', acc)
                    if acc < 0.4:
                        print('acc: ', acc)
                        print(tensor_dict[0]['study_id'])
                        print('label: ', label)
                        print('pred: ', pred_cls)
                        print('pred prob: ', pred_prob)
                        print('====' * 10)

    labels = torch.cat(labels)  # n,
    y_preds = torch.cat(y_preds, dim=0).float().softmax(dim=-1).numpy()  # n, 5

    mask = np.where(labels != -100)
    y_preds = y_preds[mask]
    labels = labels[mask]

    # calculate acc by sklean
    y_preds = np.argmax(y_preds, axis=1)
    met_ema = accuracy_score(labels, y_preds)
    val_loss = total_loss / len(dataloader)
    #xy_loss = total_xy_loss / len(dataloader)
    z_loss = total_z_loss / len(dataloader)

    print(f'val_loss:{val_loss:.6f}, '
          #f'xy_loss:{xy_loss:.6f}, '
          f'z_loss:{z_loss:.3f}, '
          f'loc_acc_ema:{met_ema:.5f}')

def infer_xy(xy_models, pred_keypoints, data_dict):
    arr_origin_list = data_dict[0]['arr_origin_list']
    nseg,_, _, _ = pred_keypoints.shape # n_seg, 2, 5, 2
    imgs = []
    for n in range(nseg):
        arr = arr_origin_list[n]
        z_len, h, w = arr.shape
        for idx in range(2):
            for level in range(5):
                z = pred_keypoints[n, idx, level, 0]
                z = int(np.round(z * z_len))
                if z < 0:
                    z = 0
                if z > z_len - 1:
                    z = z_len - 1
                imgs.append(arr[z].unsqueeze(0))
    #imgs: n_seg * 2 * 5
    imgs  = torch.cat(imgs, dim=0).unsqueeze(1)
    #print('imgs: ', imgs.shape)
    xy_pred = None
    for i in range(len(xy_models)):
        with torch.no_grad():
            with autocast:
                p = xy_models[i](imgs)
                if xy_pred is None:
                    xy_pred = p
                else:
                    xy_pred += p
    xy_pred = xy_pred / len(xy_models)
    xy_pred = xy_pred.reshape(nseg, 2, 5, 4).cpu().numpy()
    # n_seg, 2, 5, 2
    xy_pred_final = np.concatenate(
        (xy_pred[:, 0:1, :, :2], # left
        xy_pred[:, 1:2, :, 2:]), # right
        axis=1)
    xy_pred_final = xy_pred_final / 512
    # n_seg, 2, 5, 4
    pred_keypoints = np.concatenate((xy_pred_final, pred_keypoints), axis=-1)
    return pred_keypoints

if __name__ == '__main__':
    data_root = DATA_ROOT
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    #df = df[df['study_id']==114899184]

    transform = A.Compose([
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    debug_dir = f'{data_root}/debug_dir/'

    dset = Axial_Level_Dataset_Multi(data_root, df, transform=transform,
                                     with_origin_arr=True, phase='test')
    # print('dset len: ', len(dset))
    # print('df len: ', len(df))
    # exit(0)
    # study_ids = df['study_id'].unique().tolist()
    # dset = Axial_Level_Dataset_Multi_V24(data_root, study_ids, transform=transform)
    dloader = DataLoader(dset, num_workers=12, batch_size=1,
                         collate_fn=collate_fn)
    models = load_models()
    xy_models = load_xy_models()
    # val(models[0], dloader)
    # exit(0)

    study_id_to_pred_keypoints = {}

    for data_dict in tqdm.tqdm(dloader):
        study_id, series_id_list, pred_keypoints = infer_z(models, data_dict)
        pred_keypoints = infer_xy(xy_models, pred_keypoints, data_dict)
        study_id = int(study_id)
        study_id_to_pred_keypoints[study_id] = {}
        for i, sid in enumerate(series_id_list):
            sid = int(sid)
            study_id_to_pred_keypoints[study_id][sid] = {
                'points': pred_keypoints[i] # 2, 5, 4
            }
        #break
        # arr_origin_list = data_dict[0]['arr_origin_list']
        # index_info = data_dict[0]['index_info']
        # n_seg = len(index_info)
        # imgs = []
        # for n in range(n_seg):
        #     z0, z1 = index_info[n]
        #     a = arr_origin_list[n].cpu().numpy()#arr[z0:z1]
        #     z_len, h, w = a.shape
        #     for i in range(5):
        #         x = pred_keypoints[n, 0, i, 0]
        #         y = pred_keypoints[n, 0, i, 1]
        #         z = pred_keypoints[n, 0, i, 2]
        #         conf = pred_keypoints[n, 0, i, 3]
        #
        #         x = int(np.round(x * w))
        #         y = int(np.round(y * h))
        #         z = int(np.round(z * z_len))
        #         img = a[z]
        #         print('xyz: ', x, y, z, conf)
        #         img = 255 * (img - img.min()) / (img.max() - img.min())
        #         img = np.array(img, dtype=np.uint8)
        #         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #         img = cv2.circle(img, (x, y), 5, (0, 0, 255), 9)
        #         cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #         cv2.putText(img, 'level: ' + str(i), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        #                     cv2.LINE_AA)
        #         cv2.putText(img, 'sid: ' + str(n), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        #                     cv2.LINE_AA)
        #         imgs.append(img)
        #
        # img_concat = np.concatenate(imgs, axis=0)
        # cv2.imwrite(f'{debug_dir}/0_v24_axial_pred_left.jpg', img_concat)
        # exit(0)

    import pickle
    pickle.dump(study_id_to_pred_keypoints,
                open(data_root + f'/v24_axial_pred_fold0_no_ipp2.pkl', 'wb'))

    # pickle.dump(study_id_to_pred_keypoints,
    #             open(data_root + f'/v24_axial_pred_fold1_pvt.pkl', 'wb'))

    # study_id_to_pred_keypoints_origin = pickle.load(
    #     open(data_root + f'/v24_axial_pred_en5.pkl', 'rb')
    # )
    # for study_id in study_id_to_pred_keypoints.keys():
    #     print(study_id)
    #     print(study_id_to_pred_keypoints[study_id])
    #     print('=====')
    #     print(study_id_to_pred_keypoints_origin[study_id])
    #     break