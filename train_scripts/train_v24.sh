export  PYTHONPATH=../



only_val=0

CUDA_VISIBLE_DEVICES=0 python v24/pretrain_axial_cond_cls.py \
  --backbone_axial 'convnext_tiny.in12k_ft_in1k_384' \
  --bs 24 --z_imgs 3 --save_dir ./wkdir_final/v24_cond_axial/ --only_val $only_val


CUDA_VISIBLE_DEVICES=0 python v24/train_axial_cond_cls.py \
  --backbone_axial 'convnext_small.in12k_ft_in1k_384' \
  --bs 24 --z_imgs 3 --save_dir ./wkdir_final/v24_cond_axial/ --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python v24/train_axial_cond_cls.py \
  --backbone_axial 'densenet161' \
  --bs 24 --z_imgs 3 --save_dir ./wkdir_final/v24_cond_axial/ --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python v24/train_axial_cond_cls.py \
  --backbone_axial 'pvt_v2_b1.in1k' \
  --bs 24 --z_imgs 3 --save_dir ./wkdir_final/v24_cond_axial/ --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python v24/train_axial_cond_cls.py \
  --backbone_axial 'rexnetr_200.sw_in12k_ft_in1k' \
  --bs 16 --z_imgs 5 --save_dir ./wkdir_final/v24_cond_axial/ --only_val $only_val

