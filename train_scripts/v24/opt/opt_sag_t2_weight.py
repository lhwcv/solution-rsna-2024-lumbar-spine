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

data_root = '/home/hw/m2_disk/kaggle/RSNA2024_Lee/train_scripts/wkdir_final2/v20_cond_t2/'

labels = np.load(f'{data_root}/convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_666_h128_w128_with_gru/labels.npy')
weights = []
for l in labels:
    if l==0: weights.append(1)
    elif l==1: weights.append(2)
    elif l==2: weights.append(4)
    else: weights.append(0)

# scores_dict = {
# "convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h72_w128": 0.27774595915227596,
# "convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128_with_gru": 0.2781635315595207,
# "convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w128": 0.2787808083110997,
# "convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w96": 0.27911318268564866,
# #"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h128_w128": 0.27961359859717755,
# #"convnext_small.in12k_ft_in1k_384_z_imgs_3_seed_8620_h64_w128_with_gru": 0.28192255062487526,
#
# }
# preds_oof_path = []
# for k in scores_dict.keys():
#     preds_oof_path.append(f'{data_root}/{k}/final_pred_ema.npy')
preds_oof_path = sorted(glob.glob(f'{data_root}/*/final*.npy'))

oofs = [np.load(p) for p in preds_oof_path]
oofs = [p.astype(np.float64)/np.sum(p.astype(np.float64),axis=1).reshape(-1,1) for p in oofs]
preds_oof_path_name = [p.split('/')[-2] for p in preds_oof_path]

preds_dict = dict(zip(preds_oof_path_name,oofs))
preds = np.zeros((len(preds_oof_path_name), labels.shape[0],3))
for i in range(preds.shape[0]):
    preds[i] = list(preds_dict.values())[i]

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