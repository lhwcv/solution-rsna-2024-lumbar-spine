# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from scipy.linalg import norm
from math import atan
from math import sin, cos
import cv2
import pydicom
import matplotlib
import matplotlib.pyplot as plt

# shape alignment code from
# https://medium.com/@olga_kravchenko/generalized-procrustes-analysis-with-python-numpy-c571e8e8a421

REFERENCE_H = 512
REFERENCE_W = 512

level_name_to_color = {
    'L1/L2': [255, 0, 0],
    'L2/L3': [0, 255, 0],
    'L3/L4': [0, 0, 255],
    'L4/L5': [255, 255, 0],
    'L5/S1': [0, 255, 255],
}
level_color = [
    v for k, v in level_name_to_color.items()
]
level_color1 = [
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [0, 128, 128],
]


def draw_shape(
        shape,  # x,y,x,y,x,y ... format
        image=None,
        is_shift_center=True,
        is_line=True, line_thickness=1,
        is_circle=True, circle_radius=8,
        level_color=level_color
):
    if image is None:
        H, W = REFERENCE_H, REFERENCE_W
        image = np.zeros((H, W, 3))
    else:
        H, W = image.shape[:2]

    point = shape.reshape(5, 2, 2)  # assume 5x2=10 points
    if is_shift_center:
        point = point + np.array([W // 2, H // 2]).reshape(1, 1, 2)

    point = (np.round(point)).astype(np.int32)
    for i in range(5):
        color = level_color[i]
        px0, py0 = point[i, 0]
        px1, py1 = point[i, 1]
        if is_circle:
            cv2.circle(image, (px0, py0), circle_radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, (px1, py1), circle_radius, color, -1, cv2.LINE_AA)

        if is_line:
            cv2.line(image, (px0, py0), (px1, py1), color, line_thickness, cv2.LINE_AA)
            if i != 0:
                qx0, qy0 = point[i - 1, 0]
                qx1, qy1 = point[i - 1, 1]
                cv2.line(image, (px0, py0), (qx0, qy0), color, line_thickness, cv2.LINE_AA)
                cv2.line(image, (px1, py1), (qx1, qy1), color, line_thickness, cv2.LINE_AA)
    return image


def get_rotation_scale(reference_shape, shape, num_point=10):
    reference_shape = reference_shape.reshape(-1)
    shape = shape.reshape(-1)

    a = np.dot(shape, reference_shape) / norm(reference_shape) ** 2

    # separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    b = np.sum(x * ref_y - ref_x * y) / norm(reference_shape) ** 2

    scale = np.sqrt(a ** 2 + b ** 2)
    theta = atan(b / max(a, 10 ** -10))  # avoid dividing by 0

    return scale, theta


def get_rotation_matrix(theta):
    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def do_zero_mean(shape):
    shape = shape.reshape(-1, 2)
    shape = shape - shape.mean(0, keepdims=True)
    shape = shape.reshape(-1)
    return shape


def do_scale(shape, scale):
    return shape / scale


def do_rotate(shape, theta):
    mat = get_rotation_matrix(theta)
    shape = shape.reshape((-1, 2)).T
    rotated_shape = np.dot(mat, shape)
    rotated_shape = rotated_shape.T.reshape(-1)
    return rotated_shape


def do_align_shape(shape, reference_shape, is_rotate=True):
    reference_shape = np.copy(reference_shape)
    shape = np.copy(shape)

    # get scale and rotation
    scale, theta = get_rotation_scale(reference_shape, shape)

    aligned_shape = shape / scale
    if is_rotate:
        aligned_shape = do_rotate(aligned_shape, theta)

    return aligned_shape


# ---helper ---
def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype("uint8")


def read_dicom_as_image(dicom_file):
    d = pydicom.dcmread(dicom_file)
    image = d.pixel_array.astype("float32")
    image = convert_to_8bit(image)
    return image


class dotdict(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def get_saggital_t2_gt_keypoints(data_root, img_size=512):
    df = pd.read_csv(f'{data_root}/coords_rsna_improved_saggital_t2.csv')
    df = df.sort_values(['series_id', 'level'])
    print(df['level'].head(10))
    d = df.groupby("series_id")[["relative_x",
                                 "relative_y",
                                 "study_id",
                                 "series_id"
                                 ]].apply(
        lambda x: list(x.itertuples(index=False, name=None)))
    saggital_t2_gt_keypoints = {}
    all_keypoints = []
    for k, v in d.items():
        v = np.array(v)
        relative_xy = v[:, :2].reshape(5, 2, 2)
        study_id = int(v[0, 2])
        series_id = int(v[0, 3])
        keypoints = img_size * relative_xy

        if study_id not in saggital_t2_gt_keypoints.keys():
            saggital_t2_gt_keypoints[study_id] = {}
        saggital_t2_gt_keypoints[study_id][series_id] = {
            "keypoints": keypoints,
        }
        all_keypoints.append(keypoints)

    return saggital_t2_gt_keypoints, np.stack(all_keypoints)

def warp_img_and_pts(m_512, p_512, affine):

    scale, theta, shift = affine
    theta = theta / 180 * np.pi
    mat = np.array([
        scale * cos(theta), -scale * sin(theta), shift[0],
        scale * sin(theta), scale * cos(theta), shift[1],
    ]).reshape(2, 3)

    p_align = np.concatenate([p_512.reshape(-1, 2), np.ones((10, 1))], axis=1) @ mat.T
    p_align = p_align.reshape(-1, 2)
    m_align = cv2.warpAffine(m_512, mat, (512, 512))
    return m_align, p_align

if __name__ == '__main__':
    data_root = '/home/hw/m2_disk/kaggle/data/rsna-2024-lumbar-spine-degenerative-classification/'
    saggital_t2_gt_keypoints,data_shape = get_saggital_t2_gt_keypoints(data_root)

    mean = data_shape.mean(0).reshape(-1)
    zero_mean = do_zero_mean(mean)

    print('mean', mean.shape)
    print('zero_mean', zero_mean.shape)

    print('zero_mean shape\n', zero_mean)
    zero_mean = zero_mean.reshape(5, 2, 2)
    dis = []
    for i in range(4):
        p0 = zero_mean[i, 1, :]
        p1 = zero_mean[i+1, 1, :]
        dis.append(np.linalg.norm(p0 - p1))
    print('mean dis: ', np.mean(dis))
    exit(0)
 #    mean_overlay = draw_shape(
 #        zero_mean,  # x,y,x,y,x,y ... format
 #        image=None,
 #        is_shift_center=True,
 #        is_line=True, line_thickness=2,
 #        is_circle=True, circle_radius=8,
 #        level_color=level_color
 #    )
 #    plt.imshow(mean_overlay, cmap='gray')
 #    plt.show()
 #

    # compute affine parameters
    mean_512 = (zero_mean.reshape(-1, 2) + [[256, 256]]).reshape(-1)

    for study_id, v in saggital_t2_gt_keypoints.items():
        for series_id in v.keys():
            p_512 = saggital_t2_gt_keypoints[study_id][series_id]['keypoints'].copy()
            # get kaggle to mean
            # print(p_512.shape, mean_512.shape)
            mat, inlier = cv2.estimateAffinePartial2D(p_512.reshape(-1, 2), mean_512.reshape(-1, 2))

            # nomalised shape
            p_align = np.concatenate([p_512.reshape(-1, 2), np.ones((10, 1))], axis=1) @ mat.T
            # recover parameter to make normalised shape
            mat, inlier = cv2.estimateAffinePartial2D(p_512.reshape(-1, 2), p_align.reshape(-1, 2))
            scale = np.sqrt(mat[0, 0] ** 2 + mat[0, 1] ** 2)
            theta = np.arctan2(mat[1, 0,], mat[0, 0]) / np.pi * 180
            shift = mat[:, 2].tolist()

            saggital_t2_gt_keypoints[study_id][series_id]['affine'] = (scale, theta, shift)

    import pickle
    pickle.dump(saggital_t2_gt_keypoints, open(f'{data_root}/saggital_affine.pkl', 'wb'))

    if False:
        study_id = 425970461
        series_id = 8693307
        dicom_file = f'{data_root}/train_images/{study_id}/{series_id}/10.dcm'
        m = read_dicom_as_image(dicom_file)
        m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB).astype(np.uint8)
        m = cv2.resize(m, (512, 512))
        shape = saggital_t2_gt_keypoints[study_id][series_id]['keypoints'].copy()
        image = draw_shape(
            shape,  # x,y,x,y,x,y ... format
            image=m,
            is_shift_center=False,
            is_line=False, line_thickness=1,
            is_circle=True, circle_radius=8,
            level_color=level_color
        )
        plt.imshow(image, cmap='gray')
        plt.show()

        ##
        affine = saggital_t2_gt_keypoints[study_id][series_id]['affine']
        m, shape = warp_img_and_pts(m, shape, affine)
        image = draw_shape(
            shape,  # x,y,x,y,x,y ... format
            image=m,
            is_shift_center=False,
            is_line=False, line_thickness=1,
            is_circle=True, circle_radius=8,
            level_color=level_color
        )
        plt.imshow(image, cmap='gray')
        plt.show()

