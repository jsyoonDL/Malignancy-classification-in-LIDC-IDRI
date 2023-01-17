#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:39:11 2022

@author: jsyoon
"""

import torch.nn as nn
import timm
import torch
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import to_2tuple,trunc_normal_,DropPath
from timm.models.convnext import LayerNorm2d,SelectAdaptivePool2d

#%%     
class FeatureExtractor(nn.Module):
    def __init__(self,in_channel, num_features):
        super().__init__()
        # proj
        self.pool = SelectAdaptivePool2d(1,pool_type='avg')
        self.norm = LayerNorm2d(in_channel)
        self.fc = nn.Linear(in_channel,num_features)
                
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0) 


    def forward(self,x):
        x = self.pool(x)
        x = self.norm(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        
        return x   
    
#%%     
class InputFilter(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        # proj
        self.upsampling = nn.Upsample(size=(56, 56), mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.norm = LayerNorm2d(out_channel)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0) 


    def forward(self,x):
        x = self.upsampling(x)
        x = self.conv(x)
        x = self.norm(x)
        
        return x   


#%% 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
                
        
        model = timm.create_model('convnext_tiny', 
                                  pretrained=True, 
                                  num_classes=2) 
        
        self.stem = InputFilter(56,96)
        self.stages = model.stages
        self.clf = model.head
 
    def forward(self, x):   
        
        # input
        x = self.stem(x)
        x = self.stages(x)
        x_out = self.clf(x)    
     
     
        return x_out  
