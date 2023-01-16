#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:11:44 2023

@author: user
"""

import glob
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit

#%% fix random seed
seed =0
np.random.seed(seed)
random.seed(seed)
#%%

data_path = 'Data/classification/npyfiles/*/*.npy'
data_list = sorted(glob.glob(data_path))
random.shuffle(data_list)


dnum = len(data_list)
#%%

tr_num = dnum*0.5
val_num =  dnum*0.2
te_num = dnum - tr_num-val_num


indices = list(range(dnum))
te_split = te_num/dnum
#%%
sss = StratifiedShuffleSplit(n_splits=1, 
                             test_size=te_num/dnum, 
                             random_state=0)

y =  [x.split('/')[-2] for x in data_list]
for train_index, test_index in sss.split(indices, y):
    print(len(test_index), len(train_index))



sss = StratifiedShuffleSplit(n_splits=1, 
                             test_size=val_num/tr_num, 
                             random_state=0)

data_list_tr = np.array(data_list)[train_index]
y_tr =  [x.split('/')[-2] for x in data_list_tr]
for train_index, val_index in sss.split(train_index, y_tr):
    print(len(val_index), len(train_index))

split_list={
    'train': sorted(data_list_tr[train_index]),
    'val':  sorted(data_list_tr[val_index]),
    'test': sorted(np.array(data_list)[test_index])
    }


save_path = 'Data/classification/splits/'
os.makedirs(save_path, exist_ok=True)

for k in range(3):
    
    phase = list(split_list.keys())[k]
    path_info_list = split_list[phase]
    with open(save_path+phase+'.txt', mode = "w") as f:
        for info in path_info_list:
            f.write(info+'\n')
