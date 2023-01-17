#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:47:00 2023

@author: jsyoonDL
"""

import numpy as np
import matplotlib.pyplot as plt

import pylidc as pl
import glob
import pandas as pd

import os
#%%
def get_cube_from_img(img3d, center, block_size):
    """"Code for this function is based on code from this repository: https://github.com/junqiangchen/LUNA16-Lung-Nodule-Analysis-2016-Challenge"""
    # get roi(z,y,z) image and in order the out of img3d(z,y,x)range
    
    center_x = center[0]
    center_y = center[1]
    center_z = center[2]
    
    
    block_size_x = block_size[0]
    block_size_y = block_size[1]
    block_size_z = block_size[2]
    
    start_x = max(center_x - block_size_x / 2, 0)
    if start_x + block_size_x > img3d.shape[0]:
        start_x = img3d.shape[0] - block_size_x
        
        
    start_y = max(center_y - block_size_y / 2, 0)
    if start_y + block_size_y > img3d.shape[1]:
        start_y = img3d.shape[1] - block_size_y
        
        
    start_z = max(center_z - block_size_z / 2, 0)
    if start_z + block_size_z > img3d.shape[2]:
        start_z = img3d.shape[2] - block_size_z
    
    start_x = int(start_x)
    start_y = int(start_y)
    start_z = int(start_z)
    roi_img3d = img3d[ start_x:start_x + block_size_x,
                      start_y:start_y + block_size_y,
                      start_z:start_z + block_size_z]
    return roi_img3d

#%%
data_list = sorted(glob.glob('LIDC-IDRI/LIDC-IDRI*'))
data_num = len(data_list)
nodule_info = []

block_size = [64,64,56]

save_path_p = 'Data/classification/npyfiles/'
#%%
k = 0

print('------------------------------------------------')
for d_idx in range(data_num):
    
    
    pid = data_list[d_idx].split('/')[-1]
    print('processing-----{}'.format(pid))    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    vol = scan.to_volume()
    print('------------------------------------------------')
    nods = scan.cluster_annotations()
    num_nods = len(nods)
    
    #%%
    sid = scan.series_instance_uid
    
    pixel_info = scan.spacings # x,y,z
    
    for i, nod_i in enumerate(nods):
        
        
        num_name = str(k)
        if k <10 : num_name = '000' + num_name
        elif k <100: num_name = '00' + num_name
        elif k <1000: num_name = '0' + num_name
        save_name = 'Index{}.npy'.format(num_name)
        
        cent = []
        diameter = 0
        mal_factor = 0
        bbox = []
        num_ann = len(nod_i)
        for j, ann_i in enumerate(nod_i): 
            cent.append(ann_i.centroid)
            diameter += ann_i.diameter
            mal_factor += ann_i.feature_vals()[-1]
            bbox.append(ann_i.bbox_dims())
        cent = np.mean(cent,axis=0)
        diameter = diameter/num_ann
        bbox = np.max(bbox,axis=0)
        mal_factor = mal_factor/num_ann
        
        
        if mal_factor >3 : mal = 1.
        else: mal = 0.
        
        
        nodule_info.append([pid,sid,*pixel_info,*cent,*bbox,diameter,mal_factor,mal])
        patch = get_cube_from_img(vol, cent, block_size)
        
        if patch.shape[0]<block_size[0]: print('error: index {}'.format(save_name))
        
        
        if mal>0: save_path_f = save_path_p + '/True/' 
        else: save_path_f = save_path_p + '/False/'
            
        os.makedirs(save_path_f, exist_ok=True)
        np.save(save_path_f+save_name,patch)
        k +=1
#%%
column_index =['patient_id','serisuid',
               'pixel_x','pixel_y','pixel_z',
               'interp_cent_x','interp_cent_y','interp_cent_z',
               'bbox_x','bbox_y','bbox_z',
               'diameter','malignancy_level','malignancy'
               ]
nodule_info_csv = pd.DataFrame(np.array(nodule_info),columns=column_index)
nodule_info_csv = nodule_info_csv.set_index(column_index[0])
nodule_info_csv.to_csv('nodule_info.csv')
# nodule_info = pd.read_csv('nodule_info.csv')
    
