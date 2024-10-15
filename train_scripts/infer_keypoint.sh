export  PYTHONPATH=../

# infer keypoint-sag-2d + keypoint-axial-3d for stage2_model1's dataset

## infer for center slice for v2 use, now use v20 instead of early (very old) version
CUDA_VISIBLE_DEVICES=0 python v20/infer_sag_2d_keypoints.py
CUDA_VISIBLE_DEVICES=0 python v2/infer_axial_3d_keypoint.py

# infer keypoint-sag-3d + keypoint-axial-level-cls for stage2_model2's dataset
## part of keypoint-sag-3d already infered in train py v20/train_sag_3d_keypoints.py
CUDA_VISIBLE_DEVICES=0 python exps/infer_keypoint_sag_3d.py
CUDA_VISIBLE_DEVICES=0 python v24/infer_axial_level_cls.py


