#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 17:11:31 2023

@author: user
"""

import SimpleITK as sitk
import os
from os.path import isfile, join
import pandas as pd
import numpy as np
from tqdm import tqdm
from load_tools import load_itk, world_2_voxel #, show_images

#%% inital params

candidates = pd.read_csv('candidates.csv') # candodates_V2.csv (FP reduction)
num_candi = len(candidates)

width_size = 64 
depth = 24
num_subset = 10


save_path_p = 'Data/classification/npyfiles/'


#%%

for i in tqdm(range(num_candi),desc= 'canidate.npy gen'):
    im = candidates.iloc[i]
    
    num_name = str(i)
    if i <10 : num_name = '00000' + num_name
    elif i <100: num_name = '0000' + num_name
    elif i <1000: num_name = '000' + num_name
    elif i <10000: num_name = '00' + num_name
    elif i <100000: num_name = '0' + num_name
    save_name = 'Index{}.npy'.format(num_name)
    
    for subset in range(num_subset):
        input_path = "subsets/subset{}".format(subset)
        im_path = join(input_path, im['seriesuid']+'.mhd')
        if isfile(im_path): 
            lung_img = sitk.GetArrayFromImage(sitk.ReadImage(im_path))
            _, orig, spac = load_itk(im_path) # z, y, x
            vox_coords = world_2_voxel([float(im['coordZ']), float(im['coordY']), float(im['coordX'])], orig, spac)
            y_class = int(im['class'])
            w = width_size / 2
            d = depth/2
            patch = lung_img[int(vox_coords[0]-d): int(vox_coords[0]+d),
                    int(vox_coords[1] - w): int(vox_coords[1] + w),
                    int(vox_coords[2] - w): int(vox_coords[2] + w)]
            
            if y_class: save_path_f = save_path_p + '/True/' 
            else: save_path_f = save_path_p + '/False/'
                
            os.makedirs(save_path_f, exist_ok=True)
            np.save(save_path_f+save_name,patch)
