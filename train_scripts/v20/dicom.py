# -*- coding: utf-8 -*-
import os
import glob
import pydicom
import numpy as np
from skimage.transform import resize


def convert_to_8bit_lhw_version(x):
    lower, upper = np.percentile(x, (1, 99.5))
    x = np.clip(x, lower, upper)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
    return x.astype("uint8")


def load_dicom(dicom_folder,
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

