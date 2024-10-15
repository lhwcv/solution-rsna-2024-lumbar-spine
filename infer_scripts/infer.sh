export  PYTHONPATH=../


#CUDA_VISIBLE_DEVICES=0 python infer_cascade2.py

#CUDA_VISIBLE_DEVICES=0,1 python infer_cascade2_2.py

#CUDA_VISIBLE_DEVICES=0 python infer_cascade2_3.py

#CUDA_VISIBLE_DEVICES=3 python infer_cascade2_4.py


rm -r ./cache/
rm -r ./keypoints_pred/
mkdir ./cache/
mkdir ./keypoints_pred/
CUDA_VISIBLE_DEVICES=0,1 python final_infer_keypoints.py
CUDA_VISIBLE_DEVICES=0,1 python final_infer_cond.py