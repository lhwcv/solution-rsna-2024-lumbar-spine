# -*- coding: utf-8 -*-
import timm_3d
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSNA24Model_Keypoint_3D_Sag_V2(nn.Module):
    def __init__(self, model_name='densenet161', pretrained=False):
        super().__init__()
        self.model_t2 = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=2,
            num_classes=15,
            global_pool='avg'
        )
        self.model_t1 = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=2,
            num_classes=30,
            global_pool='avg'
        )

    def forward(self, x):
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        t2 = self.model_t2(x)
        t1 = self.model_t1(x)
        x = torch.cat((t2, t1), dim=-1)
        return x

class RSNA24Model_Keypoint_3D_Sag(nn.Module):
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
        #x = x.unsqueeze(1)
        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x



class RSNA24Model_Keypoint_3D_Axial(nn.Module):
    def __init__(self, model_name='densenet161', pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=1,
            num_classes=30,
            global_pool='avg'
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        return x


class RSNA24Model_Keypoint_3D_cls(nn.Module):
    def __init__(self, model_name='densenet161', in_c=30,
                 n_classes=4, pretrained=False):
        super().__init__()
        self.model = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
            in_chans=1,
            num_classes=1024,
            global_pool='avg'
        )
        self.cls_head = nn.Linear(1024, 30)
        self.reg_head = nn.Linear(1024, 30)

    def forward(self, x):
        x = x.unsqueeze(1)
        # interpolate to b, c, 96, 256, 256
        # x = F.interpolate(x, size=(96, 256, 256))
        # permute to b,c, h, w, d
        x = x.permute(0, 1, 3, 4, 2)
        x = self.model(x)
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        return reg, cls


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    x = torch.randn(1, 32, 512, 512)
    model = RSNA24Model_Keypoint_3D_Axial(model_name='convnext_small.in12k_ft_in1k_384')
    print(f'Number of parameters: {count_parameters(model) / 1e6} M')
    y = model(x)
    print(y.shape)
