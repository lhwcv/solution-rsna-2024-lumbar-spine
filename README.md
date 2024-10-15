# Kaggle Competition Solution (7th)
This is lhwcv's part (two-stage), hengck's part (one-stage) see: 
https://github.com/hengck23/solution-rsna-2024-lumbar-spine

# RSNA 2024 Lumbar Spine Degenerative Classification
- Classify lumbar spine degenerative conditions  
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification 

- For discussion, please refer to:  
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439
  
This repo contains the code to train two-stage models used in the team solution for the RSNA 2024 lumbar spine degenerative classification Kaggle competition.

## 1. Hardware  & OS
- GPU:  RTX 4090
- CPU:  20 cores, 40 threads
- Memory: 128 GB RAM
- ubuntu 18.04 LTS

## 3. Set Up Environment
- Install Python >=3.10.9
- Install requirements.txt in the python environment
- Set up the directory structure as shown below.
``` 
└── <repo_dir>
    ├── <dta> 
    |         ├── rsna-2024-lumbar-spine-degenerative-classification
    |               ├── test_images
    │               ├── train_images
    │               ├── train.csv
    │               ├── train_label_coordinates.csv
    │               ├── train_series_descriptions.csv
    │               ├── ... other files ...
    ├── .. 
    ├── LICENSE 
    ├── README.md
    ├── requirements.txt
```
- Modify the path setting by editing  "./train_scripts/data_path.py"

```
# please use the full path 
DATA_ROOT     = '... for downloaded and unzipped Kaggle data ... '
```

## 4. Set Up Dataset

- data/  contains data from "rsna-2024-lumbar-spine-degenerative-classification.zip" at:  
https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/data

- preprocess the data
```shell
sh preprocess.sh

```


## 5. Training the model


### keypoints models
Sagittal:
- **keypoint-sag-2d**: use timm model, first sample 10 slice for each series (same methods in ITK's baseline)
   then input with the center slice.
- **keypoint-sag-3d**: use timm-3d model,sagittal T2 -> 5 xyz points, T1-> 10 xyz points

Axial:
- **keypoint-axial-3d**: use timm-3d model, 10 xyz points 
- **keypoint-axial-level-cls**: 2.5d (CNN + LSTM + attention) for level classification (z), 
  and another 2d model for xy regression
  
```shell
cd train_scripts/
sh train_keypoint.sh
sh infer_keypoint.sh

```
### stage2_models1
fuse 3 view on feature space <br/>
The keypoint models used are **keypoint-sag-2d** and **keypoint-axial-3d**.  
For the sagittal plane, I sample 10 slices for each series, cropping the center 5 slices on T2, and cropping 5 slices from the left and 5 slices from the right on T1, for each level.  
For the axial plane, I sample 5 slices for each level based on the Z-coordinate of the point.  
The final input size is: x -> **(bs, 5_cond, 5_level, 5, crop_h, crop_w)**.  
The model architecture is as follows:

```
cd train_scripts/
sh train_v2.sh

```

### stage2_models2
The defect of stage2_model1 is that it uses 2D keypoints on the sagittal plane and does not handle multiple series on the axial plane very well.  
The keypoint models used are **keypoint-sag-3d** and **keypoint-axial-level-cls**.  
I crop the 3D ROI image using (z_imgs, crop_h, crop_w) based on XYZ points, where z_imgs can be either 3 or 5; I found that 3 works best for my purposes.  
For the axial plane, there may be multiple detections, but I only used the one with the highest confidence (not necessarily the best).

```
cd train_scripts/
sh train_v20.sh
sh train_v24.sh

```
### ensemble
local cv:  0.373
see infer_scripts/infer.sh

## 5. Submission csv 
Team submission notebook can be found at:  
https://www.kaggle.com/code/hengck23/lhw-v24-ensemble-add-heng
![Selection_506](https://github.com/user-attachments/assets/97cc87fa-5e4c-4897-8041-c651adea4eb0)

Team post-submission notebook can be found at:  
https://www.kaggle.com/code/hengck23/post-lhw-v24-ensemble-add-heng
![Selection_507](https://github.com/user-attachments/assets/223b40f2-11e9-4321-b231-53cb2a21ce99)

## 6. Demo
... to be updated ...

## 7. Reference trained models and validation results
https://www.kaggle.com/datasets/lihaoweicvch/lhwcv-rsna2024-final-models

## Authors

- https://www.kaggle.com/lhwcv

## License

- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
