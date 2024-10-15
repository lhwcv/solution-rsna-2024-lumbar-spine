# -*- coding: utf-8 -*-
import timm
import torch.nn as nn
import torch.nn.functional as F


#### model
class Sag_Model_25D(nn.Module):
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
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        levels = k // 5

        x = x.reshape(b * k, d, h, w)
        cond = cond.reshape(-1)
        x = self.model(x)
        cond = self.cond_emb(cond)
        cond = self.cond_fea_extractor(cond)
        x = x * cond
        x = self.drop(x)

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x


class Sag_Model_25D_no_cond(nn.Module):
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
        #         self.cond_emb = nn.Embedding(5, 64)
        #         self.cond_fea_extractor = nn.Sequential(
        #             nn.Linear(64, 128),
        #             nn.LeakyReLU(),
        #             nn.Linear(128, 256),
        #             nn.LeakyReLU(),
        #             nn.Linear(256, fea_dim),
        #             nn.Sigmoid()
        #         )
        self.drop = nn.Dropout(0.1)
        #         self.out_fea_extractor = nn.Sequential(
        #             nn.Linear(fea_dim, fea_dim),
        #             nn.LeakyReLU(),
        #             nn.Linear(fea_dim, 128)
        #         )
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 3)
        )

    def forward(self, x, cond):
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        levels = k // 5

        x = x.reshape(b * k, d, h, w)
        # cond = cond.reshape(-1)
        x = self.model(x)
        # cond = self.cond_emb(cond)
        # cond = self.cond_fea_extractor(cond)
        # x = x * cond
        x = self.drop(x)

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x


class Sag_Model_25D_Level_LSTM(nn.Module):
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
            nn.Linear(fea_dim, 3)
        )
        self.level_lstm = nn.LSTM(fea_dim,
                                  fea_dim // 2,
                                  bidirectional=True,
                                  batch_first=True, num_layers=2)

    def forward(self, x, cond):
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        n_conds = k // 5

        x = x.reshape(b * k, d, h, w)
        x = self.model(x)

        if True:
            # bs, n_conditions, 5_level, fdim
            x = x.reshape(b * n_conds, 5, -1)

            # worse with sub mean
            # xm = x.mean(dim=1, keepdims=True)
            # x = x - xm
            xm, _ = self.level_lstm(x)
            x = x + xm
            x = x.reshape(b * k, -1)

        x = self.drop(x)

        x = self.out_linear(x)
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x


class Sag_Model_25D_GRU(nn.Module):
    def __init__(self, base_model='densenet201', pool="avg",
                 in_chans=3,
                 pretrained=True):
        super(Sag_Model_25D_GRU, self).__init__()

        self.model = timm.create_model(base_model,
                                       pretrained=pretrained,
                                       num_classes=0, in_chans=1)
        nc = self.model.num_features

        self.gru = nn.GRU(nc, 512, bidirectional=True, batch_first=True, num_layers=2)
        self.exam_predictor = nn.Linear(512 * 2, 3)

        self.msd_num = 1
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for _ in range(self.msd_num)])
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_fea_extractor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x, cond=None, with_emb=False):
        # print('x shape: ', x.shape)
        bs, k, n, h, w = x.size()
        x = x.reshape(bs * k * n, 1, h, w)
        x = self.model(x)

        embeds, _ = self.gru(x.reshape(bs * k, n, x.shape[1]))
        embeds = self.pool(embeds.permute(0, 2, 1))[:, :, 0]

        if self.msd_num > 0:
            y = sum([self.exam_predictor(dropout(embeds)).reshape(bs * k, 3) for dropout in
                     self.dropouts]) / self.msd_num
        else:
            y = self.exam_predictor(embeds).reshape(bs * k, 3)

        y = y.reshape(bs, -1, 5, 3)
        y = y.reshape(bs, -1)

        if with_emb:
            embeds = embeds.reshape(bs, 3, 5, 1024)
            embeds = embeds.permute(0, 2, 1, 3).reshape(bs, 5, -1)
            embeds = self.out_fea_extractor(embeds)  # bs, 5, 512
            return y, embeds

        return y


# class Sag_Model_T1_T2(nn.Module):
#     def __init__(self, model_name='convnext_small.in12k_ft_in1k_384',
#                  pretrained=False,
#                  in_chans=3,
#                  base_model_cls=Sag_Model_25D):
#         super(Sag_Model_T1_T2, self).__init__()
#         self.sag_t1_model = base_model_cls(model_name,in_chans=in_chans,
#                                            pretrained=pretrained)
#         self.sag_t2_model = base_model_cls(model_name,in_chans=in_chans,
#                                            pretrained=pretrained)
#
#     def forward(self, x, cond):
#         if self.training:
#             return self.forward1(x, cond)
#         x_flip = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
#         pred = self.forward1(x, cond)
#         pred_flip = self.forward1(x_flip, cond)
#         return (pred + pred_flip) / 2
#
#     def forward1(self, x, cond):
#         t2 = x[:, :5]
#         t1 = x[:, 5: 15]
#         cond_t2 = cond[:, :5]
#         cond_t1 = cond[:, 5:15]
#         pred_t2 = self.sag_t2_model(t2, cond_t2)
#         pred_t1 = self.sag_t1_model(t1, cond_t1)
#         pred = torch.cat((pred_t2, pred_t1), dim=-1)
#         return pred

def build_v2_sag_model(base_model,
                       with_cond=True,
                       with_gru=False,
                       with_level_lstm=False,
                       z_imgs=3,
                       pretrained=True):
    if with_gru:
        return Sag_Model_25D_GRU(base_model, pretrained=pretrained)

    if with_level_lstm:
        return Sag_Model_25D_Level_LSTM(base_model, in_chans=z_imgs, pretrained=pretrained)

    if with_cond:
        return Sag_Model_25D(base_model, in_chans=z_imgs, pretrained=pretrained)

    return Sag_Model_25D_no_cond(base_model, in_chans=z_imgs, pretrained=pretrained)
