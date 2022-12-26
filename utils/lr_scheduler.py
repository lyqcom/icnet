"""Popular Learning Rate Schedulers"""
from __future__ import division
import math
# import torch
import mindspore.train.callback as mstracb
from bisect import bisect_right


def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr

# __all__ = ['IterationPolyLR']

# class IterationPolyLR(mstracb.LearningRateScheduler):
#     def __init__(self, optimizer, target_lr=0, max_iters=0, power=0.9, last_epoch=-1):
#         self.target_lr = target_lr
#         self.max_iters = max_iters
#         self.power = power
#         super(IterationPolyLR, self).__init__(optimizer)
#
#     def get_lr(self):
#         N = self.max_iters
#         T = self.last_epoch
#         factor = pow(1 - T / N, self.power)
#         # https://blog.csdn.net/mieleizhi0522/article/details/83113824
#         return [self.target_lr + (base_lr - self.target_lr) * factor for base_lr in self.base_lrs]

