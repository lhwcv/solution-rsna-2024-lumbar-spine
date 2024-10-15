# -*- coding: utf-8 -*-
from scipy.optimize import minimize, fsolve
import datetime
import torch.nn.functional as F
from numba import njit
import time
import pandas as pd
import numpy as np
import glob
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', UserWarning)

from sklearn.metrics import log_loss
def calc_weighted_logloss_score(y_pred):
    return log_loss(labels, y_pred, normalize=True, sample_weight=weights)

def calc_score(y_pred):
    return calc_weighted_logloss_score(y_pred)

data_root = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir_final/v20_cond_t2/'
data_root2 = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/v24/sag_t2/'

labels = np.load(f'{data_root}/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128/labels.npy')
weights = []
for l in labels:
    if l==0: weights.append(1)
    elif l==1: weights.append(2)
    elif l==2: weights.append(4)
    else: weights.append(0)

# scores_dict = {
# #data_root +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128": 1.4608310449778808e-19,
# data_root +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_legacy": 0.036734386442795405,
# data_root +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_level_lstm": 0.08371106365027185,
# data_root +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru": 0.1656687442112746,
# data_root +"convnext_small.in12k_ft_in1k_384_z_imgs_5_seed_8620_h128_w128_level_lstm": 0.13182187862852188,
# #data_root2 +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128": 2.5542532162484788e-18,
# #data_root2 +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru": 0.17528207850349836,
# data_root2 +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w128": 0.17603830128419412,
# #data_root2 +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w128_with_gru": 0.0365496627672681,
# data_root2 +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w96": 0.16676039849818616,
# #data_root2 +"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h72_w128": 0.027433486013989383,
# }



# preds_oof_path = []
# for k in scores_dict.keys():
#     preds_oof_path.append(f'{k}/final_pred_ema.npy')

w_dict = {
"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru": 0.23307717212210943,
"rexnetr_200.sw_in12k_ft_in1k_z_imgs_3_seed_8620_h128_w128_level_lstm": 0.22920191547174582,
"pvt_v2_b5.in1k_z_imgs_5_seed_8620_h128_w128_level_lstm": 0.18621063450601855,
"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w96_level_lstm": 0.12179130621594245,
"convnext_tiny.in12k_ft_in1k_384_z_imgs_5_seed_8620_h128_w128_level_lstm": 0.06665012313997162,
"convnext_tiny.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w96_level_lstm": 0.06443133164990446,
#"pvt_v2_b5.in1k_z_imgs_3_seed_8620_h72_w128_level_lstm": 0.03628244046699641,
#"pvt_v2_b1.in1k_z_imgs_3_seed_8620_h128_w128_legacy": 0.025838532963497043,
#"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w128_level_lstm": 0.017985507109863012,
#"convnext_tiny.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru": 0.010121435649156093,
}
preds_oof_path = []
for k in w_dict.keys():
    preds_oof_path.append(f'{data_root}/{k}/final_pred_ema.npy')


# preds_oof_path = sorted(glob.glob(f'{data_root}/*/final*.npy'))
# preds_oof_path_keep = []
# for p in preds_oof_path:
#     if True:#'convnext' in p:
#         preds_oof_path_keep.append(p)
# preds_oof_path = preds_oof_path_keep
#
# data_root = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir/v24/data_root2 +"'
# preds_oof_path2 = sorted(glob.glob(f'{data_root}/*/final*.npy'))
# preds_oof_path.extend(preds_oof_path2)

oofs = [np.load(p) for p in preds_oof_path]
oofs = [p.astype(np.float64)/np.sum(p.astype(np.float64),axis=1).reshape(-1,1) for p in oofs]
preds_oof_path_name = [p.split('/')[-3]+'/'+p.split('/')[-2] for p in preds_oof_path]

preds_dict = dict(zip(preds_oof_path_name,oofs))
preds = np.zeros((len(preds_oof_path_name), labels.shape[0],3))
for i in range(preds.shape[0]):
    a = list(preds_dict.values())[i]
    if a.shape[1] == 4:
        preds[i] = a[:, 1:]
    else:
        preds[i] = a

wll_scores = {}
for n, key in enumerate(preds_dict.keys()):
    score_val = calc_score(preds[n])
    wll_scores[key] = score_val
    print(f'{key:40s} Weighted_Log_Loss:', score_val)


def func_to_optimise(weights):
    pred_blend = np.tensordot(weights, preds, axes = ((0), (0)))
    score = calc_score(pred_blend)
    return score

tol = 1e-10
init_guess = [1 / preds.shape[0]] * preds.shape[0]
bnds = [(0, 1) for _ in range(preds.shape[0])]
cons = {'type': 'eq',
        'fun': lambda x: np.sum(x) - 1,
        'jac': lambda x: [1] * len(x)}

print('Inital Blend WLL:', func_to_optimise(init_guess))
start_time = time.time()

res_scipy = minimize(fun = func_to_optimise,
                     x0 = init_guess,
                     method = 'SLSQP',
                     #method='Nelder-Mead',
                     tol = tol,
                     bounds = bnds,
                     #jac = grad_func_jit,
                     constraints = cons,
                     options={"disp":True,"maxiter":1000})

print(f'[{str(datetime.timedelta(seconds = time.time() - start_time))[2:7]}] Optimised Blend Loss:', res_scipy.fun, ', Optimised Blend KLD_Loss:', func_to_optimise(res_scipy.x))
print('Optimised Weights:', res_scipy.x)
print('-' * 70)

key_weights = []
for n, key in enumerate(preds_dict.keys()):
    key_weights.append((key, res_scipy.x[n]))
    # print(f'{key:40s} Optimised Weights:', res_scipy.x[n])


key_weights = sorted(key_weights, key=lambda p: p[1], reverse=True)
for i in key_weights:
    print(f'{i[0]:40s}:', i[1])
