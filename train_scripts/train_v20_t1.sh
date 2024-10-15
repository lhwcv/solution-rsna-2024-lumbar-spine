#!/bin/bash

export  PYTHONPATH=../


# 定义backbone_sag数组
backbone_sag=("convnext_small.in12k_ft_in1k_384" "pvt_v2_b5.in1k" "pvt_v2_b2.in1k" "convnext_tiny.in12k_ft_in1k_384" "pvt_v2_b1.in1k" "rexnetr_200.sw_in12k_ft_in1k")

only_val=0

CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
  --backbone_sag "convnext_small.in12k_ft_in1k_384" \
  --z_imgs 3 --bs 32  --with_cond 1 --crop_size_h 128 --crop_size_w 128 \
  --save_dir ./wkdir_final/v20_cond_t1/ --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
  --backbone_sag "convnext_small.in12k_ft_in1k_384" \
  --z_imgs 3 --bs 32  --with_cond 0 --crop_size_h 72 --crop_size_w 128 \
  --save_dir ./wkdir_final/v20_cond_t1/ --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
  --backbone_sag "convnext_small.in12k_ft_in1k_384" \
  --z_imgs 3 --bs 32  --with_gru  1 --crop_size_h 72 --crop_size_w 128 \
  --save_dir ./wkdir_final/v20_cond_t1/ --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
  --backbone_sag "convnext_tiny.in12k_ft_in1k_384" \
  --z_imgs 3 --bs 32  --with_gru 1 --crop_size_h 128 --crop_size_w 128 \
  --save_dir ./wkdir_final/v20_cond_t1/ --only_val $only_val


# 遍历backbone_sag数组
#for backbone in "${backbone_sag[@]}"
#do
#  CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
#  --backbone_sag $backbone \
#  --z_imgs 3 --bs 32  --with_cond 1 --crop_size_h 128 --crop_size_w 128 \
#  --save_dir ./wkdir_final/v20_cond_t1/ --only_val $only_val
#
#  CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
#  --backbone_sag $backbone \
#  --z_imgs 3 --bs 32  --with_cond 0 --crop_size_h 128 --crop_size_w 128 \
#  --save_dir ./wkdir_final/v20_cond_t1/  --only_val $only_val
#
#  CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
#  --backbone_sag $backbone \
#  --z_imgs 3 --bs 32  --with_cond 0 --crop_size_h 72 --crop_size_w 128 \
#  --save_dir ./wkdir_final/v20_cond_t1/  --only_val $only_val
#
#
#  CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
#  --backbone_sag $backbone \
#  --z_imgs 3 --bs 32  --with_gru 1 --crop_size_h 128 --crop_size_w 128 \
#  --save_dir ./wkdir_final/v20_cond_t1/ --only_val $only_val
#
#
#  CUDA_VISIBLE_DEVICES=0 python exps/train_sag_t1.py \
#  --backbone_sag $backbone \
#  --z_imgs 3 --bs 32  --with_gru 1 --crop_size_h 72 --crop_size_w 128 \
#  --save_dir ./wkdir_final/v20_cond_t1/  --only_val $only_val
#
#done
