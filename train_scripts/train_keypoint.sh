export  PYTHONPATH=../

## keypoint-sag-3d

CUDA_VISIBLE_DEVICES=0 python exps/train_keypoint_sag_3d.py \
  --backbone 'densenet161' \
  --save_dir ./wkdir_final/keypoint_3d_v2_sag/

CUDA_VISIBLE_DEVICES=0 python v20/train_sag_3d_keypoints.py \
   --backbone 'densenet161' --series_description 'T1' \
   --save_dir ./wkdir_final/keypoint_3d_v20_sag_t1/ --only_val 0 --exclude_hard 0

CUDA_VISIBLE_DEVICES=0 python v20/train_sag_3d_keypoints.py \
   --backbone 'densenet161' --series_description 'T2'  \
   --save_dir ./wkdir_final/keypoint_3d_v20_sag_t2/  --only_val 0 --exclude_hard 0

CUDA_VISIBLE_DEVICES=0 python v20/train_sag_3d_keypoints.py \
  --backbone 'gluon_resnet152_v1s' --series_description 'T1' \
  --save_dir ./wkdir_final/keypoint_3d_v20_sag_t1/ --only_val 0 --exclude_hard 0

CUDA_VISIBLE_DEVICES=0 python v20/train_sag_3d_keypoints.py \
  --backbone 'gluon_resnet152_v1s' --series_description 'T2'  \
  --save_dir ./wkdir_final/keypoint_3d_v20_sag_t2/  --only_val 0 --exclude_hard 0

## keypoint-sag-2d
CUDA_VISIBLE_DEVICES=0 python v20/train_sag_2d_keypoints.py \
  --backbone 'densenet161' --series_description 'T1' \
  --save_dir ./wkdir_final/keypoint_2d_v20_sag_t1/
  
CUDA_VISIBLE_DEVICES=0 python v20/train_sag_2d_keypoints.py \
 --backbone 'fastvit_ma36.apple_dist_in1k' --series_description 'T1' \
 --save_dir ./wkdir_final/keypoint_2d_v20_sag_t1/  --base_lr 8e-4 --epochs 30 --exclude_hard 0

CUDA_VISIBLE_DEVICES=0 python v20/train_sag_2d_keypoints.py \
  --backbone 'densenet161' --series_description 'T2' \
  --save_dir ./wkdir_final/keypoint_2d_v20_sag_t2/

CUDA_VISIBLE_DEVICES=0 python v20/train_sag_2d_keypoints.py \
 --backbone 'convformer_s36.sail_in22k_ft_in1k_384' --series_description 'T2' \
 --save_dir ./wkdir_final/keypoint_2d_v20_sag_t2/  --base_lr 8e-4 --epochs 30  --exclude_hard 0

## keypoint-axial-3d
CUDA_VISIBLE_DEVICES=0 python train_keypoint_axial_3d.py \
 --backbone 'densenet161' \
 --save_dir ./wkdir_final/keypoint_3d_v2_axial/

## keypoint-axial-level-cls
CUDA_VISIBLE_DEVICES=0 python v24/train_axial_level_cls.py \
  --backbone 'convnext_small.in12k_ft_in1k_384' --save_dir \
   ./wkdir_final/keypoint_3d_v24_axial/level_cls/

CUDA_VISIBLE_DEVICES=0 python v24/train_axial_2d_keypoints.py \
  --backbone 'densenet161' \
  --save_dir ./wkdir_final/keypoint_3d_v24_axial/axial_2d_keypoints/

CUDA_VISIBLE_DEVICES=0 python v24/train_axial_2d_keypoints.py \
  --backbone 'xception65.tf_in1k' \
  --save_dir ./wkdir_final/keypoint_3d_v24_axial/axial_2d_keypoints/

