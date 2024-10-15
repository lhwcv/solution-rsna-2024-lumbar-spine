# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import timm

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