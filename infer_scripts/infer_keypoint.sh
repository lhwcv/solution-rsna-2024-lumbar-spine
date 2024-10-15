export  PYTHONPATH=../


#CUDA_VISIBLE_DEVICES=3 python infer_sag_2d_keypoint.py
#CUDA_VISIBLE_DEVICES=2 python infer_axial_3d_keypoint.py

CUDA_VISIBLE_DEVICES=0 python infer_sag_keypoint_3d_2d_cascade.py
