# -*- coding: utf-8 -*-
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

import timm_3d
from train_scripts.v24.models import Sag_Model_25D_Level_LSTM


class Axial_Level_Cls_Encoder(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=3, pretrained=True):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=3,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.lstm = nn.LSTM(fea_dim,
                            fea_dim // 2,
                            bidirectional=True,
                            batch_first=True, num_layers=2)

        self.xy_reg = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 4),
        )

        self.classifier = nn.Linear(fea_dim, 5)

    def forward(self, x, return_fea=False):
        # x = F.interpolate(x, size=(160, 160))
        # b, z_len, 256, 256
        x = F.pad(x, (0, 0, 0, 0, 1, 1))
        # 使用 unfold 函数进行划窗操作
        x = x.unfold(1, 3, 1)  # bs, z_len, 256, 256, 3
        x = x.permute(0, 1, 4, 2, 3)  # bs, z_len, 3, 256, 256
        # x = x.unsqueeze(2)  # bs, z_len, 1, 256, 256
        bs, z_len, c, h, w = x.shape
        x = x.reshape(bs * z_len, c, h, w)
        x = self.model(x).reshape(bs, z_len, -1)

        xy_pred = self.xy_reg(x)  # bs, z_len, 4
        xy_pred = xy_pred.reshape(bs, z_len, 2, 2)
        # xy_pred = xy_pred.permute(0, 2, 1, 3)  # bs, 2, z_len, 2

        x0, _ = self.lstm(x)
        x = x + x0
        out = self.classifier(x)
        if return_fea:
            return out, xy_pred, x
        return out, xy_pred


class PositionalEncoding(nn.Module):
    def __init__(self, fea_dim=512):
        super(PositionalEncoding, self).__init__()
        self.linear = nn.Linear(fea_dim, fea_dim)

    def forward(self, x, IPP_z):
        x = self.linear(x) + IPP_z.unsqueeze(-1)
        return x


class Axial_Level_Cls_Model_for_Test(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=True):
        super(Axial_Level_Cls_Model_for_Test, self).__init__()
        self.cnn_encoder = Axial_Level_Cls_Encoder(model_name, pretrained=pretrained)

        hidden_size = 512
        self.pe = PositionalEncoding(fea_dim=hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=4,
                                                   dim_feedforward=hidden_size * 4,
                                                   dropout=0.1)
        self.ts_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.level_classifier = nn.Linear(hidden_size, 5)

        self.left_classifier = nn.Linear(hidden_size, 5)
        self.right_classifier = nn.Linear(hidden_size, 5)

    def forward_arr(self, arr, IPP_z):
        level_cls_pred, xy_pred, fea = self.cnn_encoder(arr, return_fea=True)
        fea = self.pe(fea, IPP_z)
        fea =  self.ts_encoder(fea)
        level_cls_pred = self.level_classifier(fea)
        # print('level_cls_pred: ', level_cls_pred.shape)
        left_pred = self.left_classifier(fea).unsqueeze(2)
        right_pred = self.right_classifier(fea).unsqueeze(2)
        sparse_pred = torch.cat((left_pred, right_pred), dim=2)
        return level_cls_pred, sparse_pred, xy_pred

    def forward(self, d):
        arr = d[0]['arr'].unsqueeze(0)
        IPP_z = d[0]['IPP_z'].unsqueeze(0)
        level_cls_pred, sparse_pred, xy_pred = self.forward_arr(arr, IPP_z)
        if not self.training:
            x_flip1 = torch.flip(arr, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
            x_flip2 = torch.flip(arr, dims=[-2, ])  # A.VerticalFlip(p=0.5),
            level_cls_pred1, sparse_pred1, xy_pred1 = self.forward_arr(x_flip1, IPP_z)
            level_cls_pred2, sparse_pred2, xy_pred2 = self.forward_arr(x_flip2, IPP_z)
            level_cls_pred = (level_cls_pred + level_cls_pred1 + level_cls_pred2) / 3
            sparse_pred = (sparse_pred + sparse_pred1 + sparse_pred2) / 3


        coords = d[0]['coords']
        index_info = d[0]['index_info']

        return {
            'level_cls_pred': level_cls_pred,
            'sparse_pred': sparse_pred,
            'coords': coords,
            'index_info': index_info,
        }

class Axial_Level_Cls_Model(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=True):
        super(Axial_Level_Cls_Model, self).__init__()
        self.cnn_encoder = Axial_Level_Cls_Encoder(model_name, pretrained=pretrained)

        hidden_size = 512
        #self.pe = PositionalEncoding(fea_dim=hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=4,
                                                   dim_feedforward=hidden_size * 4,
                                                   dropout=0.1)
        self.ts_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.level_classifier = nn.Linear(hidden_size, 5)

        self.left_classifier = nn.Linear(hidden_size, 5)
        self.right_classifier = nn.Linear(hidden_size, 5)

    def forward_arr(self, arr, IPP_z):
        level_cls_pred, xy_pred, fea = self.cnn_encoder(arr, return_fea=True)
        #fea = self.pe(fea, IPP_z)
        fea =  self.ts_encoder(fea)
        level_cls_pred = self.level_classifier(fea)
        # print('level_cls_pred: ', level_cls_pred.shape)
        left_pred = self.left_classifier(fea).unsqueeze(2)
        right_pred = self.right_classifier(fea).unsqueeze(2)
        sparse_pred = torch.cat((left_pred, right_pred), dim=2)
        return level_cls_pred, sparse_pred, xy_pred

    def forward(self, d):
        arr = d[0]['arr'].unsqueeze(0)
        IPP_z = d[0]['IPP_z'].unsqueeze(0)
        level_cls_pred, sparse_pred, xy_pred = self.forward_arr(arr, IPP_z)
        if not self.training:
            x_flip1 = torch.flip(arr, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
            x_flip2 = torch.flip(arr, dims=[-2, ])  # A.VerticalFlip(p=0.5),
            level_cls_pred1, sparse_pred1, xy_pred1 = self.forward_arr(x_flip1, IPP_z)
            level_cls_pred2, sparse_pred2, xy_pred2 = self.forward_arr(x_flip2, IPP_z)
            level_cls_pred = (level_cls_pred + level_cls_pred1 + level_cls_pred2) / 3
            sparse_pred = (sparse_pred + sparse_pred1 + sparse_pred2) / 3
            # xy_pred = (xy_pred + xy_pred1 + xy_pred2) / 3

            # level_cls_pred = (level_cls_pred + level_cls_pred2) / 2
            # sparse_pred = (sparse_pred + sparse_pred2) / 2
            # xy_pred = (xy_pred + xy_pred2) / 2

        z_label = d[0]['label'].unsqueeze(0)
        sparse_label = d[0]['sparse_label'].unsqueeze(0)  # seq_len, 2, 5
        dense_z_mask = d[0]['dense_z_mask'].unsqueeze(0)

        xyz_keypoints = d[0]['xyz_keypoints']
        xyz_keypoints_mask = d[0]['xyz_keypoints_mask']
        coords = d[0]['coords']
        index_info = d[0]['index_info']

        return {
            'z_label': z_label,
            # 'dense_xy_keypoints': d[0]['dense_xy'].unsqueeze(0),
            # 'dense_xy_mask': d[0]['dense_xy_mask'].unsqueeze(0),
            'level_cls_pred': level_cls_pred,
            # 'xy_pred': xy_pred,
            'sparse_pred': sparse_pred,
            'sparse_label': sparse_label,
            'dense_z_mask': dense_z_mask,
            'xyz_keypoints': xyz_keypoints,
            'coords': coords,
            'index_info': index_info,
        }
        #
        # assert len(data_dict['arr_list_list']) == 1, print('current only support bs=1')
        # arr_list = data_dict['arr_list_list'][0]
        # label_list = data_dict['label_list_list'][0]
        # dense_xy_keypoints = data_dict['dense_xy_keypoints'][0]
        # dense_xy_mask = data_dict['dense_xy_mask'][0]
        #
        # arr = torch.cat(arr_list, dim=0)
        # z_label = torch.cat(label_list, dim=0).unsqueeze(0)
        # dense_xy_keypoints = torch.cat(dense_xy_keypoints, dim=1).unsqueeze(0)
        # dense_xy_mask = torch.cat(dense_xy_mask, dim=1).unsqueeze(0)
        # level_cls_pred, xy_pred = self.cnn_encoder(arr.unsqueeze(0))
        # # print('z_label: ', z_label.shape)
        # # print('dense_xy_keypoints: ', dense_xy_keypoints.shape)
        # # print('dense_xy_mask: ', dense_xy_mask.shape)
        # # print('out: ', level_cls_pred.shape)
        # # print('xy_pred: ', xy_pred.shape)
        # return {
        #     'z_label': z_label,
        #     'dense_xy_keypoints': dense_xy_keypoints,
        #     'dense_xy_mask': dense_xy_mask,
        #     'level_cls_pred': level_cls_pred,
        #     'xy_pred': xy_pred,
        # }

    def forward_debug(self, d):
        arr = d[0]['arr']
        print(arr.shape)
        level_cls_pred, xy_pred = self.cnn_encoder(arr.unsqueeze(0))
        print(level_cls_pred.shape)

        label = d[0]['label']
        sparse_label = d[0]['sparse_label']  # seq_len, 2, 5
        dense_z_mask = d[0]['dense_z_mask']
        IPP_z = d[0]['IPP_z']

        # assert len(data_dict['arr_list_list'])==1, print('current only support bs=1')
        # arr_list = data_dict['arr_list_list'][0]
        # label_list = data_dict['label_list_list'][0]
        #
        # dense_xy_keypoints = data_dict['dense_xy_keypoints'][0]
        # dense_xy_mask = data_dict['dense_xy_mask'][0]
        #
        # print(len(arr_list))
        # arr = torch.cat(arr_list, dim=0)
        # label = torch.cat(label_list, dim=0)
        # dense_xy_keypoints = torch.cat(dense_xy_keypoints, dim=1)
        # dense_xy_mask = torch.cat(dense_xy_mask, dim=1)
        # print(arr.shape)
        # print(label.shape)
        # print(dense_xy_keypoints.shape)
        # print(dense_xy_mask.shape)
        #
        # out, xy_pred = self.cnn_encoder(arr.unsqueeze(0))
        #
        #
        # print('out shape: ', out.shape)
        # print('xy_pred shape: ', xy_pred.shape)
        #
        # sparse_label_list = data_dict['sparse_label_list_list'][0]
        # keypoints_list = data_dict['keypoints_list_list'][0]
        # keypoints_mask_list = data_dict['keypoints_mask_list_list'][0]
        # IPP_xyz_list = data_dict['IPP_xyz_list_list'][0]
        '''
        # look at left
        imgs = []
        for z in range(arr.shape[0]):
            if dense_xy_mask[0, z,0] !=0:
                x = int(dense_xy_keypoints[0, z, 0])
                y = int(dense_xy_keypoints[0, z, 1])
                print('xyz: ', x, y, z)
                img = arr[z].numpy()
                img = 255 * (img - img.min()) / (img.max() - img.min())
                img = np.array(img, dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.circle(img, (x, y), 5, (0, 0, 255), 3)
                cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                la = int(label[z])
                cv2.putText(img, 'level: '+str(la), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                imgs.append(img)

        debug_dir = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/debug_dir'
        img_concat = np.concatenate(imgs, axis=0)
        cv2.imwrite(f'{debug_dir}/0_v24_axial_data_in_model_left.jpg', img_concat)
        #print(dense_xy_mask[0])
        #

        # look at right
        imgs = []
        for z in range(arr.shape[0]):

            if dense_xy_mask[1, z, 0] != 0:
                x = int(dense_xy_keypoints[1, z, 0])
                y = int(dense_xy_keypoints[1, z, 1])
                print('xyz: ', x, y, z)
                img = arr[z].numpy()
                img = 255 * (img - img.min()) / (img.max() - img.min())
                img = np.array(img, dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.circle(img, (x, y), 5, (0, 0, 255), 3)
                cv2.putText(img, str(z), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                la = int(label[z])
                cv2.putText(img, 'level: ' + str(la), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                imgs.append(img)

        debug_dir = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/debug_dir'
        img_concat = np.concatenate(imgs, axis=0)
        cv2.imwrite(f'{debug_dir}/0_v24_axial_data_in_model_right.jpg', img_concat)
        '''


class Axial_Model_25D(nn.Module):
    def __init__(self, model_name='densenet201', in_chans=3,
                 n_classes=4, pretrained=False):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.drop = nn.Dropout(0.1)
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 6)
        )

    def forward(self, x):
        # 5 level: b, 10, d, h, w
        b, k, d, h, w = x.shape

        n_conds = k // 5

        x = x.reshape(b * k, d, h, w)
        x = self.model(x)

        x = self.drop(x)

        x = self.out_linear(x)
        x = x.reshape(b, k, 6)
        left_x = x[:, :k // 2, :3]
        right_x = x[:, k // 2:, 3:]

        # b, k, 3
        x = torch.cat((left_x, right_x), dim=1)

        x = x.reshape(b, -1)
        return x


class Axial_Model_25D_GRU(nn.Module):
    def __init__(self, base_model='densenet201', in_chans=3, pretrained=True):
        super(Axial_Model_25D_GRU, self).__init__()

        self.model = timm.create_model(base_model,
                                       pretrained=pretrained,
                                       num_classes=0, in_chans=in_chans)
        nc = self.model.num_features

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, 3)

        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_fea_extractor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, axial_imgs, with_emb=False):
        bs, k, n, c, h, w = axial_imgs.size()
        axial_imgs = axial_imgs.reshape(bs * k * n, c, h, w)
        x = self.model(axial_imgs)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 3) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 3)

        y = y.reshape(bs, 2, 5, 3)
        y = y.reshape(bs, -1)

        if with_emb:
            # bs, 2_cond, 5_level, 1024
            embeds = embeds.reshape(bs, 2, 5, 1024)
            embeds = self.out_fea_extractor(embeds)  # bs, 2, 5, 128
            return y, embeds

        return y


class Axial_HybridModel_24(nn.Module):
    def __init__(self, backbone_sag='densenet201',
                 backbone_axial='densenet201',
                 pretrained=False,
                 axial_in_channels=3):
        super().__init__()
        self.sag_model = Sag_Model_25D_Level_LSTM(backbone_sag,
                                                  in_chans=4,
                                                  pretrained=pretrained,
                                                  with_emb=True,
                                                  with_level_lstm=True)
        self.axial_model = Axial_Model_25D_GRU(base_model=backbone_axial,
                                               in_chans=axial_in_channels,
                                               pretrained=pretrained)
        fdim = 2 * 128
        self.out_linear = nn.Linear(fdim, 3)

    def forward(self, s_t2, axial_x):
        if self.training:
            return self.forward_train(s_t2, axial_x)
        else:
            return self.forward_test(s_t2, axial_x)

    def forward_train(self, s_t2, axial_x):
        _, sag_emb = self.sag_model.forward(s_t2)
        bs = axial_x.shape[0]

        axial_pred, axial_emb = self.axial_model.forward(axial_x, with_emb=True)
        # bs, 2_cond, 5_level, 3
        axial_pred = axial_pred.reshape(bs, 2, 5, 3)

        ### fuse ####
        # bs, 2_cond, 5_level, fdim
        fea = torch.cat((sag_emb, axial_emb), dim=-1)
        # bs, 2_cond, 5_level, 3
        ys = self.out_linear(fea)# + axial_pred
        ys = ys.reshape(bs, -1)
        return ys

    def forward_test(self, x, axial_x):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),
        ys1 = self.forward_train(x, axial_x)
        ys2 = self.forward_train(x_flip, axial_x_flip)
        ys = (ys1 + ys2) / 2
        return ys


class Axial_HybridModel_24_sag3d(nn.Module):
    def __init__(self, backbone_sag='densenet201',
                 backbone_axial='densenet201',
                 pretrained=False):
        super().__init__()
        self.sag_model = timm_3d.create_model(
            "densenet161",
            pretrained=pretrained,
            features_only=False,
            in_chans=1,
            num_classes=256,
            global_pool='avg'
        )
        self.axial_model = Axial_Model_25D_GRU(base_model=backbone_axial,
                                               pretrained=pretrained)
        fdim = 256 + 2 * 128
        self.out_linear = nn.Linear(fdim, 6)
        self.level_lstm = nn.LSTM(fdim,
                                  fdim // 2,
                                  bidirectional=True,
                                  batch_first=True, num_layers=2)

    def forward(self, s_t2, axial_x):
        if self.training:
            return self.forward_train(s_t2, axial_x)
        else:
            return self.forward_test(s_t2, axial_x)

    def forward_train(self, s_t2, axial_x):

        # s_t2 b, 5, 16, h, w
        bs, _, d, h, w = s_t2.shape
        s_t2 = s_t2.reshape(bs * 5, 1, d, h, w)
        # permute to b,c, h, w, d
        s_t2 = s_t2.permute(0, 1, 3, 4, 2)
        # bs, 5, fdim
        sag_emb = self.sag_model(s_t2).reshape(bs, 5, -1)

        axial_pred, axial_emb = self.axial_model.forward(axial_x, with_emb=True)
        # bs, 2_cond, 5_level, 3
        axial_pred = axial_pred.reshape(bs, 2, 5, 3)

        ### fuse ####
        # bs, 2_cond, 5_level, fdim
        axial_emb = axial_emb.permute(0, 2, 1, 3)
        axial_emb = axial_emb.reshape(bs, 5, -1)
        fea = torch.cat((sag_emb, axial_emb), dim=-1)
        xm, _ = self.level_lstm(fea)
        fea = fea + xm

        # bs, 5_level, 2_cond, 3
        ys = self.out_linear(fea).reshape(bs, 5, 2, 3)
        # bs, 2_cond, 5_level, 3
        ys = ys.permute(0, 2, 1, 3)
        ys = (ys + axial_pred) / 2
        ys = ys.reshape(bs, -1)
        return ys

    def forward_test(self, x, axial_x):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        axial_x_flip = torch.flip(axial_x, dims=[-2])  # A.VerticalFlip(p=0.5),
        ys1 = self.forward_train(x, axial_x)
        ys2 = self.forward_train(x_flip, axial_x_flip)
        ys = (ys1 + ys2) / 2
        return ys

class Axial_Model_25D_GRU_cls6(nn.Module):
    def __init__(self, base_model='densenet201', pool="avg", pretrained=True):
        super(Axial_Model_25D_GRU_cls6, self).__init__()

        self.model = timm.create_model(base_model,
                                       pretrained=pretrained,
                                       num_classes=0, in_chans=3)
        nc = self.model.num_features

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, 6)

        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_fea_extractor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, axial_imgs, cond=None, with_emb=False):
        bs, k, n, c, h, w = axial_imgs.size()
        axial_imgs = axial_imgs.reshape(bs * k * n, 3, h, w)
        x = self.model(axial_imgs)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 6) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 6)

        y = y.reshape(bs, 5, 2, 3)
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(bs, -1)

        if with_emb:
            embeds = embeds.reshape(bs, 2, 5, 1024)
            embeds = embeds.permute(0, 2, 1, 3).reshape(bs, 5, -1)
            embeds = self.out_fea_extractor(embeds)  # bs, 5, 512
            return y, embeds

        return y


class Axial_Cond_Cls_Encoder(nn.Module):
    def __init__(self,
                 model_name='densenet201',
                 pretrained=True):
        super().__init__()
        fea_dim = 512
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=1,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.lstm = nn.LSTM(fea_dim,
                            fea_dim // 2,
                            bidirectional=True,
                            batch_first=True, num_layers=2)
        self.classifier = nn.Linear(fea_dim, 6)

    def forward_train(self, x):
        x = x.unsqueeze(2)  # bs, z_len, 1, 256, 256
        bs, z_len, c, h, w = x.shape
        x = x.reshape(bs * z_len, c, h, w)
        x = self.model(x).reshape(bs, z_len, -1)
        x0, _ = self.lstm(x)
        x = x + x0
        out = self.classifier(x)
        return out, x

    def forward(self, x, level_prob=None, return_fea=False):
        """

        :param x:
        :param level_prob: (bs, z_len, 5)
        :return:
        """
        x = x.unsqueeze(2)  # bs, z_len, 1, 256, 256
        bs, z_len, c, h, w = x.shape
        x = x.reshape(bs * z_len, c, h, w)
        x = self.model(x).reshape(bs, z_len, -1)
        x0, _ = self.lstm(x)
        fea = x + x0
        if level_prob is None:
            return fea

        x = fea.unsqueeze(1)  # bs, 1, z_len, 512
        level_prob = level_prob.permute(0, 2, 1).unsqueeze(-1)  # bs, 5, z_len, 1
        x = (level_prob * x).sum(dim=2) / (1e-3 + level_prob.sum(dim=2))  # bs, 5, 512
        x = self.classifier(x).reshape(-1, 5, 2, 3)  # bs, 5, 2, 3
        x = x.permute(0, 2, 1, 3)  # bs, 2_cond, 5_level, 3
        bs = x.shape[0]
        cond_pred = x.reshape(bs, -1)
        if return_fea:
            return cond_pred, fea
        return cond_pred
