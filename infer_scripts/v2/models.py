# -*- coding: utf-8 -*-
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm_3d


class Model_iafoss(nn.Module):
    def __init__(self, base_model='densenet201', pool="avg", pretrain=True):
        super(Model_iafoss, self).__init__()
        self.base_model = base_model
        NUM_CLS = 15
        self.model = timm.create_model(self.base_model, pretrained=pretrain, num_classes=0, in_chans=3)
        nc = self.model.num_features
        # self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(nc, 512),
        #                           nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLS))

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, NUM_CLS)
        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input1):
        shape = input1.size()
        batch_size = shape[0]
        n = shape[1]

        input1 = input1.reshape(-1, shape[2], shape[3], shape[4])
        x = self.model(input1)

        embeds, _ = self.gru(x.reshape(batch_size, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(batch_size, 5, -1) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(batch_size, 5, -1)

        # """
        embeds = embeds.reshape(batch_size, 1, 1024)
        return y, embeds


#### model
class RSNA24Model(nn.Module):
    def __init__(self, model_name='densenet201', in_c=30,
                 n_classes=4, pretrained=False,fea_dim=1024):
        super().__init__()
        fea_dim = fea_dim
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=5,
            num_classes=fea_dim,
            global_pool='avg'
        )
        self.cond_emb = nn.Embedding(5, 64)
        self.cond_fea_extractor = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, fea_dim),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(0.1)
        self.out_fea_extractor = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 128)
        )
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 3)
        )

    def forward(self, x, cond):
        b, k, d, h, w = x.shape
        x = x.reshape(b * k, d, h, w)
        cond = cond.reshape(-1)
        x = self.model(x)
        cond = self.cond_emb(cond)
        cond = self.cond_fea_extractor(cond)
        x = x * cond
        x = self.drop(x)
        fea = self.out_fea_extractor(x)

        # bs, 5_cond, 5_level, 128
        fea = fea.reshape(b, 5, 5, 128)

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x, fea


class HybridModel_V2(nn.Module):
    def __init__(self,
                 model_name1='densenet201',
                 model_name2='densenet161',
                 fea_dim=512,
                 pretrained=False):
        super().__init__()
        self.sag_model = RSNA24Model(model_name1, pretrained=pretrained,
                                     fea_dim=fea_dim)
        self.axial_model = Model_iafoss(base_model=model_name2, pretrain=pretrained)
        fdim = 1024 + 5 * 128
        self.out_linear = nn.Linear(fdim, 15)

    def forward_axial(self, axial_x, axial_x_flip):
        # b, 5, 6, 3, h, w
        b, k, d, c, h, w = axial_x.shape
        axial_embs = []
        axial_embs_flip = []
        for ik in range(k):
            input1 = axial_x[:, ik]
            _, emb = self.axial_model(input1)
            axial_embs.append(emb)

            input1 = axial_x_flip[:, ik]
            _, emb_flip = self.axial_model(input1)
            axial_embs_flip.append(emb_flip)

        # bs, 5_level, 1024
        axial_embs = torch.cat(axial_embs, dim=1)
        axial_embs_flip = torch.cat(axial_embs_flip, dim=1)
        return  axial_embs, axial_embs_flip

    def forward(self, x, axial_x, cond, n_list):
        x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        axial_x_flip_list = []  # A.VerticalFlip(p=0.5),
        b = x.shape[0]
        axial_x_list = []
        for i in range(b):
            #print(n_list[i])
            axial_x_list.append(axial_x[i, :, :n_list[i]])

        for i in range(len(axial_x_list)):
            axial_x_flip_list.append( torch.flip(axial_x_list[i], dims=[-2]) )

        all_axial_embs = []
        all_axial_embs_flip = []
        for i in range(len(axial_x_list)):
            axial_x = axial_x_list[i].unsqueeze(0)
            axial_x_flip = axial_x_flip_list[i].unsqueeze(0)
            axial_embs, axial_embs_flip = self.forward_axial(axial_x, axial_x_flip)
            all_axial_embs.append(axial_embs)
            all_axial_embs_flip.append(axial_embs_flip)

        axial_embs = torch.cat(all_axial_embs, dim=0)
        axial_embs_flip = torch.cat(all_axial_embs_flip, dim=0)



        ys2, sag_embs = self.sag_model(x, cond)
        ys2 = ys2.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        ys2_flip, sag_embs_flip = self.sag_model(x_flip, cond)
        ys2_flip = ys2_flip.reshape(b, 5, 5, 3)  # bs, 5_cond, 5_level, 3

        # sag_embs: bs, 5_cond, 5_level, 128
        sag_embs = sag_embs.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128
        sag_embs_flip = sag_embs_flip.permute(0, 2, 1, 3).reshape(b, 5, -1)  # bs, 5_level, 5*128

        #
        embs = torch.cat((sag_embs, axial_embs), dim=-1)
        out1 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        out1 = out1.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        out1 = out1 + ys2

        embs = torch.cat((sag_embs_flip, axial_embs_flip), dim=-1)
        out4 = self.out_linear(embs).reshape(b, 5, 5, 3)  # bs, 5_level, 5_cond, 3
        out4 = out4.permute(0, 2, 1, 3)  # # bs, 5_cond, 5_level, 3
        out4 = out4 + ys2_flip

        ys = (out1 + out4) / 2.0
        ys = ys.reshape(b, -1)
        return ys

class RSNA24Model_Keypoint_3D_Sag_V2(nn.Module):
    def __init__(self, model_name='densenet161', pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=2,
            num_classes=45,
            global_pool='avg'
        )

    def forward(self, x):
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x