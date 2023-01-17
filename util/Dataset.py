#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:04:04 2023

@author: jsyoonDL
"""

from torch.utils.data import Dataset
import glob
import torch
import numpy as np
#%% Custom dataset train
class Dataset(Dataset):
    def __init__(self, path, mode ='train'):
        super().__init__() 
        
       
        with open('{}/splits/{}.txt'.format(path,mode), 'r') as fin:
            data_list = [line.replace('\n','') for line in fin]
            
        
        self.img_path_list = sorted(data_list)
        self.label_list =  [x.split('/')[-2] for x in self.img_path_list]

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        label = int(self.label_list[index]=='True')
        
        img = np.load(img_path).astype(float)
        img = (img - img.min()) / (img.max() - img.min())
        img = torch.Tensor(img.transpose(2,1,0))


        return img, label


    def __len__(self):
        return len(self.img_path_list)
    
    def get_labels(self):
        return self.label_list
