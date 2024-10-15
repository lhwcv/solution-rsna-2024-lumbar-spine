# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import timm_3d
from timm_3d.layers.classifier import create_classifier
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
import cv2

_backbone_fea_channels = {
    'convnext_small.in12k_ft_in1k_384': [96, 192, 384, 768],
    'densenet201': [64, 256, 512, 1792, 1920],
    'densenet161': [96, 384, 768, 2112, 2208],
    'densenet121': [64, 256, 512, 1024, 1024],
}


class FPN_3D_Head4(nn.Module):
    def __init__(self,
                 channels=[64, 256, 512, 1024],
                 out_c=256):
        super(FPN_3D_Head4, self).__init__()
        self.conv1 = nn.Conv3d(channels[0], out_c, kernel_size=(1, 1, 1))
        self.conv2 = nn.Conv3d(channels[1], out_c, kernel_size=(1, 1, 1))
        self.conv3 = nn.Conv3d(channels[2], out_c, kernel_size=(1, 1, 1))
        self.conv4 = nn.Conv3d(channels[3], out_c, kernel_size=(1, 1, 1))

    def _upsample_add(self, x, y):
        _, _, H, W, D = y.size()
        return F.interpolate(x, size=(H, W, D), mode='trilinear') + y

    def forward(self, x1, x2, x3, x4):
        p4 = self.conv4(x4)
        p3 = self._upsample_add(p4, self.conv3(x3))
        p2 = self._upsample_add(p3, self.conv2(x2))
        p1 = self._upsample_add(p2, self.conv1(x1))

        return p1, p2, p3, p4


class ROI_align_25D(nn.Module):
    def __init__(self,
                 xy_crop_size=None,
                 xy_crop_ratio=1 / 8.0,
                 output_size=(2, 2)):
        super().__init__()
        self.xy_crop_size = xy_crop_size
        self.xy_crop_ratio = xy_crop_ratio
        self.output_size = output_size

    def forward(self, fea, keypoints):
        '''

        :param fea: e.g  b, c, 128, 128, 8
        :param keypoints: b, 5, 3,  range (0, 1)
        :return:
        '''
        b, c, h, w, depth = fea.shape
        keypoints[:, :, 0] = w * keypoints[:, :, 0]
        keypoints[:, :, 1] = h * keypoints[:, :, 1]
        #keypoints[:, :, 2] = depth * keypoints[:, :, 2]

        xy_crop_size = self.xy_crop_size
        if xy_crop_size is None:
            xy_crop_size = int(h * self.xy_crop_ratio)

        xmin = keypoints[:, :, 0] - xy_crop_size
        xmax = keypoints[:, :, 0] + xy_crop_size

        xmin = xmin.clamp(min=0, max=w - 1)
        xmax = xmax.clamp(min=0, max=w - 1)

        ymin = keypoints[:, :, 1] - xy_crop_size
        ymax = keypoints[:, :, 1] + xy_crop_size

        ymin = ymin.clamp(min=0, max=h - 1)
        ymax = ymax.clamp(min=0, max=h - 1)

        # Prepare batches of rois
        rois = torch.zeros((b * 5, 5), device=fea.device)
        for i in range(b):
            rois[i * 5: (i + 1) * 5, 0] = i
            rois[i * 5: (i + 1) * 5, 1] = xmin[i]
            rois[i * 5: (i + 1) * 5, 2] = ymin[i]
            rois[i * 5: (i + 1) * 5, 3] = xmax[i]
            rois[i * 5: (i + 1) * 5, 4] = ymax[i]

        feas = []
        for d in range(depth):
            aligned = roi_align(fea[:, :, :, :, d], rois, self.output_size)
            _, c, oH, oW = aligned.shape
            aligned = aligned.reshape(b, 5, c, oH, oW).unsqueeze(-1)
            feas.append(aligned)
        # b, 5, c, oH, oW, depth
        feas = torch.cat(feas, dim=-1)
        return feas



class Sag_Classify_Fea(nn.Module):
    def __init__(self, xy_crop_ratio=1 / 8.0,
                 roi_align_out_size=(2, 2),
                 channels=128,
                 fdim=512):
        super(Sag_Classify_Fea, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels,
                      kernel_size=(3, 3, 1),
                      padding=(1, 1, 0)),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(),
        )
        self.roi_pool = ROI_align_25D(xy_crop_ratio=xy_crop_ratio,
                                      output_size=roi_align_out_size)
        d = roi_align_out_size[0] * roi_align_out_size[1]
        # Spinal Canal Stenosis
        self.linear_scs = nn.Linear(channels * d, fdim)
        # Neural Foraminal Narrowing
        self.linear_nfn = nn.Linear(channels * d, fdim)

    def forward(self, x, keypoints):
        '''

        :param x: b, c, h, w, d
        :return:
        '''
        x = self.conv1(x)
        # b, 5, c, oH, oW, depth
        x = self.roi_pool(x, keypoints)
        b, _, c, oH, oW, depth = x.shape
        x = x.reshape(b, 5, c * oH * oW, depth)
        x = x.permute(0, 1, 3, 2)
        scs_emb = self.linear_scs(x)
        nfn_emb = self.linear_nfn(x)
        right_emb = nfn_emb[:, :, :depth // 2, :]
        left_emb = nfn_emb[:, :, depth // 2:, :]
        # mean on depth
        scs_emb = scs_emb.mean(dim=2)
        left_emb = left_emb.mean(dim=2)
        right_emb = right_emb.mean(dim=2)
        return scs_emb, left_emb, right_emb


class Sag_Model3D(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm_3d.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=2,
            global_pool='none',
        )
        num_features = _backbone_fea_channels[model_name][-1]
        if 'densenet' in model_name:
            channels = np.array(_backbone_fea_channels[model_name])[1:].tolist()
            self.fpn = FPN_3D_Head4(channels, out_c=64)
        else:
            channels = _backbone_fea_channels[model_name]
            self.fpn = FPN_3D_Head4(channels, out_c=64)

        self.reg_pool, self.point_reg = create_classifier(
            num_features,
            num_classes=15,
            pool_type='avg', )

        fdim = 512
        self.p1_fea = Sag_Classify_Fea(xy_crop_ratio=1 / 16.0,
                 roi_align_out_size=(4, 4),
                 channels=64,
                 fdim=fdim)
        # self.p2_fea = Sag_Classify_Fea(fdim=fdim)
        # self.p3_fea = Sag_Classify_Fea(fdim=fdim)

        # self.scs_classifier = nn.Linear(fdim, 3)
        # self.nfn_classifier = nn.Linear(fdim, 3)
        # self.ss_classifier = nn.Sequential(
        #     nn.Linear(3 * fdim, fdim),
        #     nn.LeakyReLU(),
        #     nn.Linear(fdim, 6)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(3 * fdim, fdim),
            nn.LeakyReLU(),
            nn.Linear(fdim, 15)
        )

    def forward(self, x, keypoints=None):
        if 'densenet' in self.model_name:
            _, x1, x2, x3, x4 = self.backbone(x)
        else:
            x1, x2, x3, x4 = self.backbone(x)
        if keypoints is None:
            pred_keypoints = self.point_reg(self.reg_pool(x4)).reshape(-1, 5, 3)
        else:
            pred_keypoints = keypoints

        p1, p2, p3, p4 = self.fpn(x1, x2, x3, x4)
        #print('p1 shape: ', p1.shape)

        scs_emb, left_emb, right_emb = self.p1_fea(p1, pred_keypoints)
        #scs_emb_p1, left_emb_p1, right_emb_p1 = self.p1_fea(p1, pred_keypoints)
        # scs_emb_p2, left_emb_p2, right_emb_p2 = self.p2_fea(p2, pred_keypoints)
        # scs_emb_p3, left_emb_p3, right_emb_p3 = self.p3_fea(p3, pred_keypoints)
        # scs_emb = scs_emb_p1 + scs_emb_p2 + scs_emb_p3
        # left_emb = left_emb_p1 + left_emb_p2 + left_emb_p3
        # right_emb = right_emb_p1 + right_emb_p2 + right_emb_p3
        emb_all = torch.cat((scs_emb, left_emb, right_emb), dim=2)

        # scs_pred = self.scs_classifier(scs_emb)
        # left_nfn_pred = self.nfn_classifier(left_emb)
        # right_nfn_pred = self.nfn_classifier(right_emb)
        # ss_pred = self.ss_classifier(emb_all)
        #
        # pred = torch.cat((scs_pred, left_nfn_pred, right_nfn_pred, ss_pred), dim=-1)
        pred = self.classifier(emb_all)
        #print(pred.shape)

        # b, level_5, cond_5, 3
        pred = pred.reshape(-1, 5, 5, 3)
        # b, cond_5, level_5,  3
        pred = pred.permute(0, 2, 1, 3)
        # print('pred shape: ', pred.shape)

        pred = pred[:, :3, :, :]
        bs = pred.shape[0]
        pred = pred.reshape(bs, -1)
        if keypoints is None:
            return pred, pred_keypoints
        return pred


def gen_test_img():
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    img2 = np.zeros((512, 512, 3), dtype=np.uint8)
    step = 80
    keypoints = []
    for i in range(5):
        x = 256
        y = step * (i + 1)
        keypoints.append([x / 512.0, y / 512.0, 0.5])
        text = str(i)
        cv2.circle(img1, (x, y), 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img1, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        text = str(5 - i)
        cv2.circle(img2, (x, y), 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img2, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return img1, img2, np.array(keypoints)


def test_roi_align():
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'

    img1, img2, p = gen_test_img()
    cv2.imwrite(f'{data_root}/img1.png', img1)
    cv2.imwrite(f'{data_root}/img2.png', img2)

    layer = ROI_align_25D(xy_crop_size=32, output_size=(64, 64))
    fea = torch.zeros(2, 3, 512, 512, 1)
    fea[0] = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(-1).float()

    fea[1] = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(-1).float()
    keypoints = torch.ones(2, 5, 3)
    keypoints[0] = torch.from_numpy(p)
    keypoints[1] = torch.from_numpy(p)
    feas = layer(fea, keypoints)
    for b in range(2):
        imgs = feas[b]
        for i in range(5):
            im = imgs[i][:, :, :, 0].permute(1, 2, 0)
            print(im.max())
            im = im.numpy()
            im = 255 * (im - im.min()) / (im.max() - im.min())
            im = im.astype(np.uint8)
            print(im.shape)
            cv2.imwrite(f'{data_root}/img{b}_{i}.png', im)


if __name__ == '__main__':
    # test_roi_align()
    x = torch.randn(1, 2, 512, 512, 32)
    model = Sag_Model3D(model_name='convnext_small.in12k_ft_in1k_384')
    model(x)
