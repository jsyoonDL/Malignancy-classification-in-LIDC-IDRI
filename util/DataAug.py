#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 21:50:47 2022

@author: jsyoon
"""
import kornia.augmentation as K
import torch.nn as nn
from torch import Tensor
import torch


#%% DataAugmentation
class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.transforms = nn.Sequential(
            # K.RandomEqualize(p=0.75),
            # K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomThinPlateSpline(p=0.5),
            K.RandomAffine((0,180),p=0.5),
            K.RandomPerspective(0.5,p=0.5),
            # K.RandomElasticTransform(p=0.5),            
            # K.RandomGaussianNoise(mean=0,std=0.05, p=0.5),
            # K.RandomSharpness(sharpness=0.25, p=0.5),
            # K.RandomCrop((80, 80), p=1., cropping_mode="resample")
            )
        # self.cutmix = K.RandomCutMixV2(p=0.5)
        # self.enhance = kornia.enhance.equalize_clahe()
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        # x = self.enhance(x)
        x_out = self.transforms(x)  # BxCxHxW
        # x_out = self.cutmix(x_out)
        return x_out