
# lhwcv part

First, I would like to express my gratitude to Kaggle and RSNA for hosting this great competition.<br/>
I'm also grateful to my teammates, I learned a lot from them. <br/>

## Summary
2 stage methods, components:
- **stage1**: keypoint regression (both 2d and 3d)
- **stage2_model1**: crop by keypoint, fuse 3 view on feature space
- **stage2_model2**: crop by keypoint, single view for 3 types of condition
- **ensemble**: various backbone + flip TTA

## stage1
keypoint regression (both 2d and 3d) <br/>

Sagittal:
- **keypoint-sag-2d**: use timm model, first sample 10 slice for each series (same methods in ITK's baseline)
   then input with the center slice.
- **keypoint-sag-3d**: use timm-3d model,sagittal T2 -> 5 xyz points, T1-> 10 xyz points

Axial:
- **keypoint-axial-3d**: use timm-3d model, 10 xyz points 
- **keypoint-axial-level-cls**: 2.5d (CNN + LSTM + attention) for level classification (z), 
  and another 2d model for xy regression
  
## stage2_model1
fuse 3 view on feature space <br/>
The keypoint models used are **keypoint-sag-2d** and **keypoint-axial-3d**.  
For the sagittal plane, I sample 10 slices for each series, cropping the center 5 slices on T2, and cropping 5 slices from the left and 5 slices from the right on T1, for each level.  
For the axial plane, I sample 5 slices for each level based on the Z-coordinate of the point.  
The final input size is: x -> **(bs, 5_cond, 5_level, 5, crop_h, crop_w)**.  
The model architecture is as follows:

```python
# train 
y, axial_embs = self.axial_model(axial_x) # bs, 
ys2, sag_embs = self.sag_model(x)
# sag_embs: bs, 5_cond, 1_level, 128
sag_embs = sag_embs.permute(0, 2, 1, 3).reshape(b, 1, -1)  # bs, 1, 5*128
embs = torch.cat((sag_embs, axial_embs), dim=-1)
ys = self.out_linear(embs).reshape(b, 1, 5, 3)  # bs, 1_level, 5_cond, 3
ys = ys.permute(0, 2, 1, 3)  # # bs, 5_cond, 1_level, 3
ys = ys + ys2
```

## stage2_model2
The defect of stage2_model1 is that it uses 2D keypoints on the sagittal plane and does not handle multiple series on the axial plane very well.  
The keypoint models used are **keypoint-sag-3d** and **keypoint-axial-level-cls**.  
I crop the 3D ROI image using (z_imgs, crop_h, crop_w) based on XYZ points, where z_imgs can be either 3 or 5; I found that 3 works best for my purposes.  
For the axial plane, there may be multiple detections, but I only used the one with the highest confidence (not necessarily the best).


## ensemble
stage2_model1 + stage2_model2 with many backbones, TTA: hflip for sagittal, vflip for axial <br/>
**results**: cv: 0.373  LB: 0.35  PB: 0.40

**backbones**: 
- pvt-b1
- pvt-b2
- convnext-tiny
- convnext-small
- densenet161


# hecgck & lhwcv ensemble
more details to be out …
LB: 0.34 PB: 0.40

## Code
We will release it soon after we finish cleaning and consolidating.



# hecgck part
see  https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/539439