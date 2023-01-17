#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:46:00 2023

@author: jsyoonDL
"""

import torch
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR

from tqdm import tqdm
import os

from util.Dataset import Dataset
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from torch.utils.data import Subset
from util.DataAug import DataAugmentation
import numpy
import random

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#%%
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seedTrue
    os.environ['PYTHONHASHSEED'] = str(seed)
#%% train
def train(model, params):
    
    
    #Parsing params
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    optimizer = params['optimizer']
    loss_function=params['loss_function']
    data_path = params['data_path']
    model_path = params['model_path']
    norm = params['norm']
    l_lambda = params['lambda']
    best =0
    ds_tr = Dataset(data_path,'train')
    ds_val = Dataset(data_path,'val')
    
    
    dl_tr = DataLoader(ds_tr, 
                       batch_size=batch_size, 
                       pin_memory=True,
                       shuffle = True,
                       num_workers=0)
   
    dl_val = DataLoader(ds_val,
                       batch_size=batch_size, 
                       pin_memory=True,
                       shuffle = False,
                       num_workers=0)
    dl = {'train':dl_tr, 'val': dl_val}
    
    
    augmentation = DataAugmentation()
    
    
    # scheduler = StepLR(optimizer, step_size=len(dl_tr)*20, gamma=0.75)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dl_tr), eta_min=0, last_epoch=-1)

    #%% training      
    for epoch in range(0, num_epochs):
        with tqdm(dl['train'], unit="batch") as tepoch:
            total = 0
            correct = 0
            # idx = 0
            
            model.train()
            for data in tepoch:
                # tepoch.set_description(f"Epoch {epoch}") # progress bar
                tepoch.set_description(f"LR {optimizer.param_groups[0]['lr']},Epoch {epoch}") # progress bar
               
                inputs, labels = data  # data assign
                inputs = augmentation(inputs) # data augmenation           
                inputs = inputs.cuda()         
                labels = labels.cuda()
                
                # Batch initialization
                optimizer.zero_grad()     
                
                # forward + back propagation 
                outputs = model(inputs)
                
                predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                accuracy = 100*correct / total      
                
                train_loss = loss_function(outputs, labels)
                
                # regularization p=2 L2 p=1 L1
                
                if norm == 1:
                    # l_lambda = 1e-3
                    l_norm = torch.norm(torch.cat([p.view(-1) for p in model.parameters()]), p=norm)
                    train_loss = train_loss+l_lambda*l_norm
                elif norm ==2: 
                    # l_lambda = 1e-3
                    l_norm = torch.norm(torch.cat([p.view(-1) for p in model.parameters()]), p=norm)
                    train_loss = train_loss+l_lambda*l_norm
                    
                
                
                
                train_loss.backward()
                optimizer.step()
                
                scheduler.step() # iter
                # Display current eval.
                tepoch.set_postfix(loss=train_loss.item(),accuracy=accuracy)
#%% val             

            if (epoch==0) or epoch>15*num_epochs//20: 
                total = 0
                correct = 0
                test_loss = 0
                accuracy = []                    
                loss =[]
                # Initialize the prediction and label lists(tensors)
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(dl['val'], 0):
                            inputs, labels = data
                            inputs = inputs.cuda()       
                            labels = labels.cuda()      
                          
                            outputs = model(inputs)
                        
                            _, predicted = torch.max(outputs, 1)    
                            
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            
                            #loss
                            test_loss = loss_function(outputs, labels)  
                            test_loss += test_loss.item()   
                accuracy.append(100 * correct/total)
                loss.append(100 *test_loss/total)
                
                if best<=100*correct/total:
                # if best>=100 *test_loss/total:
                    save_path = model_path
                    os.makedirs(save_path, exist_ok=True)                    
                    # torch.save(model.module.state_dict(), save_path + '/trained_model.pt')
                    torch.save(model.state_dict(), save_path + '/trained_model.pt')
                    best = 100*correct/total                  
                    
                    
                # display the test results
                # print('Epoch: %d/%d, Tr.loss: %.6f, Val.loss: %.6f, Val.Acc.: %.2f, Best loss.: %.2f'
                #   %(epoch+1, num_epochs, train_loss.item(), 100 *test_loss/total, 100*correct/total,loss_pre))
                print('Epoch: %d/%d, Tr.loss: %.6f, Val.loss: %.6f, Val.Acc.: %.2f, Best Acc.: %.2f'
                  %(epoch+1, num_epochs, train_loss.item(), 100 *test_loss/total, 100*correct/total,best))
    return best
#%%    
# out_time = time.time()
# pro_time = out_time-in_time 
# print(pro_time)
