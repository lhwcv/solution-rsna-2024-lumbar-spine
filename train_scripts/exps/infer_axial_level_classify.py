# -*- coding: utf-8 -*-
import os
import timm
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pickle
import pandas as pd
import torch.nn.functional as F
from src.utils.dicom import convert_to_8bit_lhw_version
from src.utils.comm import create_dir
import json
import pydicom
import glob
from skimage.transform import resize
from src.utils.dicom import convert_to_8bit_lhw_version
import torch.nn as nn
import albumentations as A


# window_center , window_width, intercept, slope = get_windowing(data)
# data.pixel_array = data.pixel_array * slope + intercept
# min_value = window_center - window_width // 2
# max_value = window_center + window_width // 2
# data.pixel_array.clip(min_value, max_value, out=data.pixel_array)

def load_dicom_axial(dicom_folder,
                     plane='axial',
                     reverse_sort=True,
                     img_size=512):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicom_instance_numbers = [int(i.split('/')[-1][:-4]) for i in dicom_files]

    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)

    PatientPosition = dicoms[0].PatientPosition
    # dicom_instance_numbers = np.array(dicom_instance_numbers)
    # reverse_sort = False
    # if PatientPosition in ['FFS', 'FFP']:
    #     reverse_sort = True
    # idx = np.argsort(-dicom_instance_numbers if reverse_sort else dicom_instance_numbers)

    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    dicom_instance_numbers = np.array(dicom_instance_numbers)[idx]

    cols = np.asarray([d.pixel_array.shape[1] for d in dicoms]).astype("int")[idx].tolist()
    rows = np.asarray([d.pixel_array.shape[0] for d in dicoms]).astype("int")[idx].tolist()
    instance_num_to_shape = {}
    dicom_instance_numbers = dicom_instance_numbers.tolist()
    for i, n in enumerate(dicom_instance_numbers):
        instance_num_to_shape[n] = [cols[i], rows[i]]

    array = []
    for i, d in enumerate(dicoms):
        arr = d.pixel_array.astype("float32")
        arr = resize(arr, (img_size, img_size))
        array.append(arr)
    array = np.array(array)
    array = array[idx]
    array = convert_to_8bit_lhw_version(array)

    PixelSpacing = np.asarray([d.PixelSpacing for d in dicoms]).astype("float")[idx]
    SliceThickness = np.asarray([d.SliceThickness for d in dicoms]).astype("float")[idx]
    SpacingBetweenSlices = np.asarray([d.SpacingBetweenSlices for d in dicoms]).astype("float")[idx]
    SliceLocation = np.asarray([d.SliceLocation for d in dicoms]).astype("float")[idx]

    dicom_instance_numbers_to_idx = {}
    for i, ins in enumerate(dicom_instance_numbers):
        dicom_instance_numbers_to_idx[ins] = i
    meta = {
        "ImagePositionPatient": ipp,
        'SliceLocation': SliceLocation,
        "PixelSpacing": PixelSpacing,
        "SliceThickness": SliceThickness,
        "SpacingBetweenSlices": SpacingBetweenSlices,
        "PatientPosition": PatientPosition,

        "instance_num_to_shape": instance_num_to_shape,
        "dicom_instance_numbers": dicom_instance_numbers,
        "dicom_instance_numbers_to_idx": dicom_instance_numbers_to_idx,
    }
    return array, meta


def rescale_keypoints_by_meta(keypoints, meta, img_size=512):
    dicom_instance_numbers_to_idx = meta['dicom_instance_numbers_to_idx']
    instance_num_to_shape = meta['instance_num_to_shape']
    rescaled_keypoints = -np.ones_like(keypoints)
    for i in range(len(keypoints)):
        x, y, ins_num = keypoints[i]
        # no labeled
        if x < 0:
            continue
        origin_w, origin_h = instance_num_to_shape[ins_num]
        x = img_size / origin_w * x
        y = img_size / origin_h * y
        z = dicom_instance_numbers_to_idx[ins_num]
        rescaled_keypoints[i, 0] = x
        rescaled_keypoints[i, 1] = y
        rescaled_keypoints[i, 2] = z
    return rescaled_keypoints


def _get_keypoint_mean_margin(pts):
    # pts 5, 3
    mean_margin = []
    for i in range(4):
        z0 = pts[i, 2]
        z1 = pts[i + 1, 2]
        if z0 != -1 and z1 != -1:
            z = abs(z1 - z0)
            mean_margin.append(z)
    if len(mean_margin) == 0:
        mean_margin = -1
    else:
        mean_margin = np.mean(mean_margin)
    return mean_margin


def _gen_label_by_keypoints(pts, mean_margin, z_len,
                            label=None, sparse_label=None):
    # pts (5, 3)
    if label is None:
        label = -100 * np.ones(z_len)
    if sparse_label is None:
        sparse_label = np.zeros((z_len, 5), dtype=np.float32)

    margin = min(mean_margin / 4, 2)
    # margin = 2
    for level, p in enumerate(pts):
        z = p[2]
        if z < 0:
            assert z == -1
            continue
        start_idx = int(np.round(z - margin))
        if start_idx < 0:
            start_idx = 0
        end_idx = int(np.round(z + margin)) + 1
        if end_idx > z_len:
            end_idx = z_len
        # if level == 0:
        #     print(start_idx, end_idx)
        label[start_idx:end_idx] = level
        d = max(end_idx - z, z - start_idx)
        assert d != 0
        slope = 0.5 / d

        for i in range(start_idx, end_idx):
            dis = abs(i - z)
            s = 1.0 - dis * slope
            sparse_label[i, level] = s

    return label, sparse_label


def check_label(label_):
    label = label_[label_ != -100].reshape(-1).tolist()
    label_unique = []
    for la in label:
        if la not in label_unique:
            if len(label_unique) > 0:
                if la < label_unique[len(label_unique) - 1]:
                    return False
            label_unique.append(la)
    return True


class Axial_Level_Dataset(Dataset):
    def __init__(self, data_dir, df: pd.DataFrame, transform=None):
        super(Axial_Level_Dataset, self).__init__()
        desc_df = pd.read_csv(f"{data_dir}/train_series_descriptions.csv")
        self.image_dir = f"{data_dir}/train_images/"
        self.desc_df = desc_df[desc_df['series_description'].isin(['Axial T2'])]
        self.coord_label_df = pd.read_csv(f"{data_dir}/train_label_coordinates.csv")
        level2int = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        self.coord_label_df = self.coord_label_df.replace(level2int)
        self.df = df
        self.transform = transform
        self.samples = []
        hard_axial_series_id_list = [1771893480,
                                     3014494618,
                                     2114122300,
                                     2714421343,
                                     2693084890,
                                     2444210364
                                     ]
        for _, row in df.iterrows():
            study_id = row['study_id']
            label = row[1:].values.astype(np.int64)
            label = label[15:].reshape(2, 5)
            g = self.desc_df[self.desc_df['study_id'] == study_id]
            series_id_list = g['series_id'].to_list()
            if len(series_id_list) != 1:
                continue

            for series_id in series_id_list:
                if series_id in hard_axial_series_id_list:
                    continue
                coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
                coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
                if len(coord_sub_df) > 0:
                    self.samples.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'label': label
                    })

        print('samples: ', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def get_axial_coord(self, study_id, series_id):
        coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
        coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
        keypoints = -np.ones((5, 2, 3), np.float32)  # 5_level, left/right, xyz

        for _, row in coord_sub_df.iterrows():
            idx = 0
            if row['condition'] == 'Right Subarticular Stenosis':
                idx = 1
            x, y, z = row['x'], row['y'], row['instance_number']

            keypoints[row['level'], idx, 0] = x
            keypoints[row['level'], idx, 1] = y
            keypoints[row['level'], idx, 2] = z

        keypoints = keypoints.reshape(-1, 3)
        return keypoints

    def __getitem__(self, idx):
        item = self.samples[idx]
        study_id = item['study_id']
        series_id = item['series_id']
        cls_label = item['label']
        img_size = 320
        keypoints = self.get_axial_coord(study_id, series_id)
        # print(keypoints)
        dicom_dir = f'{self.image_dir}/{study_id}/{series_id}/'
        arr, meta = load_dicom_axial(dicom_dir, img_size=512)

        PatientPosition = meta['PatientPosition']
        if PatientPosition == "FFS":
            arr = np.flip(arr, -1)
            arr = np.ascontiguousarray(arr)

        arr = arr.transpose(1, 2, 0)
        arr = A.center_crop(arr, 384, 384)
        arr = arr.transpose(2, 0, 1)

        keypoints = rescale_keypoints_by_meta(keypoints, meta, img_size=img_size)
        # print('rescaled: ', keypoints)
        # print('PatientPosition: ', meta['PatientPosition'])
        # print('arr shape: ', arr.shape)
        # resample by SpacingBetweenSlices
        SpacingBetweenSlices = meta['SpacingBetweenSlices'][0]
        target_spacing = 3.0
        z_scale = SpacingBetweenSlices / target_spacing
        depth = int(arr.shape[0] * z_scale)

        # target_size = (depth, img_size, img_size)
        target_size = (depth, img_size, img_size)
        arr = F.interpolate(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0),
                            size=target_size).numpy().squeeze()

        # keypoints[:, 2] = keypoints[:, 2] * z_scale
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] != -1:
                keypoints[i, 2] = keypoints[i, 2] * z_scale

        keypoints = keypoints.reshape(5, 2, 3)
        left_keypoints = keypoints[:, 0]
        right_keypoints = keypoints[:, 1]
        mean_margin_left = _get_keypoint_mean_margin(left_keypoints)
        mean_margin_right = _get_keypoint_mean_margin(right_keypoints)

        z_len = arr.shape[0]
        if mean_margin_left != -1 and mean_margin_right != -1:
            mean_margin = (mean_margin_right + mean_margin_left) / 2
        elif mean_margin_left == -1 and mean_margin_right != -1:
            mean_margin = mean_margin_right
        elif mean_margin_right == -1 and mean_margin_left != -1:
            mean_margin = mean_margin_left
        else:
            mean_margin = 9 * target_spacing / 4.0

        label, sparse_label_left = _gen_label_by_keypoints(left_keypoints,
                                                           mean_margin, z_len)
        label, sparse_label_right = _gen_label_by_keypoints(right_keypoints,
                                                            mean_margin, z_len, label)
        sparse_label = (sparse_label_left + sparse_label_right) / 2
        if self.transform is not None:
            arr = arr.transpose(1, 2, 0)
            arr = self.transform(image=arr)['image']
            arr = arr.transpose(2, 0, 1)
        # print(study_id, label)
        if not check_label(label):
            print('!!!!!')
            print('study_id: ', study_id)
            print('sid: ', series_id)
            coord_sub_df = self.coord_label_df[self.coord_label_df['study_id'] == study_id]
            coord_sub_df = coord_sub_df[coord_sub_df['series_id'] == series_id]
            for _, row in coord_sub_df.iterrows():
                print(row)
            print('label: ', label)
            print('keypoints: ', keypoints)

        for i in range(5):
            if left_keypoints[i, 2] < 0:
                cls_label[0, i] = -100
            if right_keypoints[i, 2] < 0:
                cls_label[1, i] = -100
        cls_label = cls_label.reshape(-1)

        return {
            'img': arr.astype(np.float32),
            'label': label.astype(np.int64),
            'sparse_label': sparse_label,
            'cls_label': cls_label,

            'study_id': study_id,
            'series_id': series_id,
            'PatientPosition': meta['PatientPosition']
        }


def test_dataset():
    import tqdm

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # df = df[df['study_id']==26342422]
    dset = Axial_Level_Dataset(data_root, df)
    dloader = DataLoader(dset, num_workers=0, batch_size=1)
    lens = []
    PatientPosition_list = []
    for d in tqdm.tqdm(dloader):
        print(d['sparse_label'])
        exit(0)
        arr = d['img']
        lens.append(arr[0].shape[0])
        PatientPosition = d['PatientPosition'][0]
        PatientPosition_list.append(PatientPosition)

    print(pd.DataFrame(lens).describe())
    print(pd.DataFrame(PatientPosition_list).describe())
    print(pd.DataFrame(PatientPosition_list).value_counts())


class Axial_Level_Cls_Model(nn.Module):
    def __init__(self, model_name='densenet201',
                 in_chans=3, pretrained=True):
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
        self.lstm = nn.LSTM(fea_dim,
                            fea_dim // 2,
                            bidirectional=True,
                            batch_first=True, num_layers=2)
        self.classifier = nn.Linear(fea_dim, 5)

    def forward_test(self, x):
        x_flip1 = torch.flip(x, dims=[-1, ])  # A.HorizontalFlip(p=0.5),
        x_flip2 = torch.flip(x, dims=[-2, ])  # A.VerticalFlip(p=0.5),
        x0 = self.forward_train(x)
        x1 = self.forward_train(x_flip1)
        x2 = self.forward_train(x_flip2)
        return (x0 + x1 + x2) / 3

    def forward_train(self, x):
        # b, z_len, 256, 236
        x = F.pad(x, (0, 0, 0, 0, 1, 1))
        # 使用 unfold 函数进行划窗操作
        x = x.unfold(1, 3, 1)  # bs, z_len, 256, 236, 3
        x = x.permute(0, 1, 4, 2, 3)  # bs, z_len, 3, 256, 236
        bs, z_len, c, h, w = x.shape
        x = x.reshape(bs * z_len, c, h, w)
        x = self.model(x).reshape(bs, z_len, -1)
        x0, _ = self.lstm(x)
        x = x + x0
        x = self.classifier(x)
        return x

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_test(x)


if __name__ == '__main__':
    test_dataset()
    exit(0)
    import tqdm

    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    df = pd.read_csv(f'{data_root}/train.csv')
    df = df.fillna(-100)
    label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
    df = df.replace(label2id)
    # df = df[df['study_id']==26342422]
    transforms_val = A.Compose([
        A.Normalize(mean=0.5, std=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    dset = Axial_Level_Dataset(data_root, df, transform=transforms_val)
    valid_dl = DataLoader(dset, num_workers=12, batch_size=1)
    model = Axial_Level_Cls_Model("convnext_small.in12k_ft_in1k_384", pretrained=False)
    model.cuda()
    model_dir = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/cls/axial_level3_only_1seg/convnext_small.in12k_ft_in1k_384_lr_0.0002/'
    model.load_state_dict(
        torch.load(f'{model_dir}/best_fold_0_ema.pt'))
    model.eval()
    autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.half)

    y_preds = []
    labels = []
    with tqdm.tqdm(valid_dl, leave=True) as pbar:
        with torch.no_grad():
            for idx, tensor_dict in enumerate(pbar):

                img = tensor_dict['img'].cuda()
                label = tensor_dict['label'].cuda()

                with autocast:
                    pred = model(img)
                    pred = pred.cpu().reshape(-1, 5)
                    label = label.cpu().reshape(-1)

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
                    print(tensor_dict['study_id'][0])
                    print(tensor_dict['series_id'][0])
                    print(img.shape)
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
    acc = accuracy_score(labels, y_preds)
    print('acc: ', acc)
