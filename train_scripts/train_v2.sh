export  PYTHONPATH=../

only_val=0


CUDA_VISIBLE_DEVICES=0 python train_cls_v2_all_view.py \
  --backbone_sag 'convnext_small.in12k_ft_in1k_384' \
  --backbone_axial 'densenet161' \
  --axial_crop_with_model 0 \
  --only_val 0 --axial_crop_size 256 --axial_margin_extend 1.0 \
  --sag_img_size 128 --save_dir ./wkdir_final/v2_cond/  --only_val $only_val


CUDA_VISIBLE_DEVICES=0 python train_cls_v2_all_view.py \
  --backbone_sag 'pvt_v2_b1.in1k' \
  --backbone_axial 'pvt_v2_b1.in1k' \
  --axial_crop_with_model 0 \
  --only_val 0 --axial_crop_size 256 --axial_margin_extend 1.0 \
  --sag_img_size 128 --save_dir ./wkdir_final/v2_cond/  --only_val $only_val


CUDA_VISIBLE_DEVICES=0 python train_cls_v2_all_view.py \
  --backbone_sag 'pvt_v2_b1.in1k' \
  --backbone_axial 'densenet161' \
  --axial_crop_with_model 0 \
  --only_val 0 --axial_crop_size 256 --axial_margin_extend 1.0 \
  --sag_img_size 128 --save_dir ./wkdir_final/v2_cond/  --only_val $only_val

CUDA_VISIBLE_DEVICES=0 python train_cls_v2_all_view.py \
  --backbone_sag 'pvt_v2_b2.in1k' \
  --backbone_axial 'pvt_v2_b2.in1k' \
  --axial_crop_with_model 0 \
  --only_val 0 --axial_crop_size 256 --axial_margin_extend 1.0 \
  --sag_img_size 128 --save_dir ./wkdir_final/v2_cond/  --only_val $only_val


CUDA_VISIBLE_DEVICES=0 python train_cls_v2_all_view.py \
  --backbone_sag 'convnext_nano.in12k_ft_in1k' \
  --backbone_axial 'convnext_nano.in12k_ft_in1k' \
  --axial_crop_with_model 0 \
  --only_val 0 --axial_crop_size 256 --axial_margin_extend 1.0 \
  --sag_img_size 128 --save_dir ./wkdir_final/v2_cond/  --only_val $only_val
