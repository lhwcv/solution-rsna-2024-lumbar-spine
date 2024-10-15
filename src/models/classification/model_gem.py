# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import numpy as np

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


class Sag_Model_25D_GEM2(nn.Module):
    def __init__(self, model_name='convnext_small.in12k_ft_in1k_384', in_chans=3,
                 n_classes=4, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            global_pool='none'
        )
        self.pool = GeM()
        fea_dim = 768
        self.out_linear = nn.Sequential(
            nn.Linear(fea_dim, fea_dim),
            nn.LeakyReLU(),
            nn.Linear(fea_dim, 3)
        )
        self.drop = nn.Dropout(0.1)


    def forward(self, x, cond):
        # 1 level : b, 5, d, h, w
        # 5 level: b, 25, d, h, w
        b, k, d, h, w = x.shape

        levels = k // 5

        x = x.reshape(b * k, d, h, w)
        _,_, _, x = self.model(x)
        x = self.pool(x)[:, :, 0, 0]
        x = self.out_linear(self.drop(x))
        x = x.reshape(b, k, 3)
        x = x.reshape(b, -1)
        return x


if __name__ == '__main__':
    b = 1
    d = 3
    h = 128
    w = 128
    x = torch.randn(b, 5, d, h, w)
    model = Sag_Model_25D_GEM2(in_chans=d)
    model(x, None)