# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import timm

import timm_3d


class RSNA24Model_Keypoint_2D(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=1, pretrained=False,
                 num_classes=10):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool='avg'
        )

    def forward(self, x):
        x = self.model(x)
        return x

class RSNA24Model_Keypoint_3D(nn.Module):
    def __init__(self, model_name='densenet161',
                 in_chans=1,
                 num_classes=30,
                 pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool='avg'
        )

    def forward(self, x):

        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x