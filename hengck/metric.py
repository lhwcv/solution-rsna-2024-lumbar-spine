# -*- coding: utf-8 -*-
import numpy as np
import sklearn


def do_eval(probability, truth):
    # probability : N,2,5,3    N x condition x level x grade
    # truth: N,2,5

    p = probability.reshape(-1, 3)
    t = truth.reshape(-1)
    # truth = np.eye(3)[truth] #one-hot

    loss = []
    count = []
    for i in [0, 1, 2]:  # 3 grade
        l = -np.log(p[t == i][:, i])
        L = len(l)
        if L == 0:
            count.append(0)
            loss.append(0)
        else:
            count.append(L)
            loss.append(l.mean())

    weight = [1, 2, 4]  # [1,1,1] #=
    weighted_loss = (
            (weight[0] * count[0] * loss[0] + weight[1] * count[1] * loss[1] + weight[2] * count[2] * loss[2]) /
            (weight[0] * count[0] + weight[1] * count[1] + weight[2] * count[2])
    )

    # ---
    any_loss = 0
    if 0:
        any_truth = truth.reshape(-1, 5, 2)
        any_prob = probability.reshape(-1, 5, 2, 3)

        any_truth = (any_truth.reshape(-1, 10) == 2).max(-1)
        any_prob = (any_prob.reshape(-1, 10, 3)[..., 2]).max(-1)
        any_weight = (any_truth == 1) * 4 + (any_truth != 1) * 1
        any_loss = sklearn.metrics.log_loss(
            y_true=any_truth,
            y_pred=any_prob,
            sample_weight=any_weight,
        )

    valid_loss = [
        weighted_loss, loss[0], loss[1], loss[2], any_loss
    ]
    return valid_loss


def loss_message(valid_loss):
    weight,  mild, modr, sevr, any_loss = valid_loss
    text = f'weight: {weight:5.3f}   '
    text += f'mild: {mild:5.3f}   '
    text += f'modr: {modr:5.3f}   '
    text += f'sevr: {sevr:5.3f}   '
    text += f'any_loss: {any_loss:5.3f}   '
    return text
