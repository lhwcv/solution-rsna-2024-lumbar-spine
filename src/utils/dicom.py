# -*- coding: utf-8 -*-
import os
import cv2
import glob
import numpy as np
import pydicom
from skimage.transform import resize


def read_dicom_itk_version(src_path=None, dicom_data=None):
    if dicom_data is None:
        dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    return image.astype("uint8")


def convert_to_8bit_vaillant_version(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype("uint8")


def convert_to_8bit_lhw_version(x):
    lower, upper = np.percentile(x, (1, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
    return x.astype("uint8")


def resize_volume(x, w=640, h=640):
    xs = []
    for i in range(len(x)):
        img = x[i]
        img = cv2.resize(img, (w, h),interpolation=cv2.INTER_CUBIC)
        xs.append(img)
    return np.array(xs, dtype=np.uint8)

def load_dicom_stack_vaillant_version(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    array = array[idx]
    return {"array": convert_to_8bit_vaillant_version(array),
            "positions": ipp,
            "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float")}


def load_dicom_stack_lhw_version(dicom_folder, plane, reverse_sort=False,
                                 img_size = 512):
    dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicom_instance_numbers = [int(i.split('/')[-1][:-4]) for i in dicom_files]

    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    dicom_instance_numbers = np.array(dicom_instance_numbers)[idx]

    cols = np.asarray([d.pixel_array.shape[1] for d in dicoms]).astype("int")[idx].tolist()
    rows = np.asarray([d.pixel_array.shape[0] for d in dicoms]).astype("int")[idx].tolist()
    instance_num_to_shape = {}
    dicom_instance_numbers = dicom_instance_numbers.tolist()
    for i, n in enumerate(dicom_instance_numbers):
        instance_num_to_shape[n] = [cols[i], rows[i]]

    # try:
    #     array = np.stack([d.pixel_array.astype("float32") for d in dicoms])
    # except Exception as e:
    #     print(e)
    #     print('\n now force align with the first dicom\n')
    #     target_shape = dicoms[0].pixel_array.shape
    #     target_PixelSpacing = dicoms[0].PixelSpacing
    #     target_SpacingBetweenSlices = dicoms[0].SpacingBetweenSlices
    #     array = []
    #     for i, d in enumerate(dicoms):
    #
    #         if i == 0:
    #             array.append(d.pixel_array.astype("float32"))
    #             continue
    #         assert d.SpacingBetweenSlices == target_SpacingBetweenSlices
    #         scale0 = d.PixelSpacing[0] / target_PixelSpacing[0]
    #         scale1 = d.PixelSpacing[1] / target_PixelSpacing[1]
    #         arr = d.pixel_array.astype("float32")
    #         output_shape = (round(arr.shape[0] * scale0),
    #                         round(arr.shape[1] * scale1))
    #         assert target_shape == output_shape
    #         arr = resize(arr, output_shape)
    #         array.append(arr)
    #     array = np.array(array)
    #
    # array = array[idx]
    # array = convert_to_8bit_lhw_version(array)
    # array = resize_volume(array, 512, 512)

    ##check
    # ps = np.asarray(dicoms[0].PixelSpacing).astype("float")
    # for d in dicoms:
    #     p = np.asarray(d.PixelSpacing).astype("float")
    #     err = (ps - p).sum()
    #     assert err ==0, print(ps, p)

    # array = []
    # for i, d in enumerate(dicoms):
    #     arr = d.pixel_array.astype("float32")
    #     arr = resize(arr, (img_size, img_size))
    #     array.append(arr)
    # array = np.array(array)
    # array = array[idx]
    # array = convert_to_8bit_lhw_version(array)

    PixelSpacing = np.asarray([d.PixelSpacing for d in dicoms]).astype("float")[idx]
    SliceThickness = np.asarray([d.SliceThickness for d in dicoms]).astype("float")[idx]
    SpacingBetweenSlices = np.asarray([d.SpacingBetweenSlices for d in dicoms]).astype("float")[idx]
    SliceLocation = np.asarray([d.SliceLocation for d in dicoms]).astype("float")[idx]
    PatientPosition = dicoms[0].PatientPosition
    return {
            #"array": array,
            "positions": ipp,
            'SliceLocation': SliceLocation,
            "instance_num_to_shape": instance_num_to_shape,
            "dicom_instance_numbers": dicom_instance_numbers,
            "PixelSpacing": PixelSpacing,
            "SliceThickness": SliceThickness,
            "SpacingBetweenSlices": SpacingBetweenSlices,
            "PatientPosition": PatientPosition
            }
