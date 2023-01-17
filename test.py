#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:04:37 2023

@author: jsyoonDL
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from timm.scheduler.cosine_lr import CosineLRScheduler

from tqdm import tqdm
import os
import time

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from util.Dataset import Dataset

import random
from sklearn.metrics import f1_score
from torchmetrics.functional import auroc,precision_recall_curve, auroc,auc
from torchmetrics.functional import specificity, precision_recall
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics

from model.Model import Model



import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#%%
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)  

#%%
def metric_report(y_true,y_pred):
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)    
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    # Report = [TPR,TNR,PPV,NPV,FPR,FNR,FDR,ACC]
    # Report = pd.DataFrame(Report,index = ['TPR','TNR','PPV','NPV','FPR','FNR','FDR','ACC'])
    
    Report = [TPR,TNR,PPV,ACC]
    Report = pd.DataFrame(Report,index = ['Sensitivity','Specificity','Precision','ACC'])
    
    return Report.T
#%% test
def test(model, params):
    #Parsing params
    batch_size = params['batch_size']
    loss_function=params['loss_function']
    device=params['device']
    data_path = params['data_path']
    model_name = params['model_name']
    
    ds = Dataset(data_path,'test')
    classes = ['False','True']
   
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        pin_memory=True,
        shuffle=False, 
        num_workers=4) 
           
    total = 0
    correct = 0
    accuracy = []                    
    
    # Initialize the prediction and label lists(tensors)
    pred_list=torch.zeros(0,dtype=torch.long, device='cpu')
    lb_list=torch.zeros(0,dtype=torch.long, device='cpu')
    
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dl, 0),desc='test_results'): 
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
          
            outputs = model(inputs)
        
            # _, predicted = torch.max(outputs, 1)
            predict_proba = torch.nn.Softmax(dim=-1)(outputs)
            _, predicted = torch.max(predict_proba, 1)
            # Append batch prediction results
            pred_list=torch.cat([pred_list,predicted.view(-1).cpu()])        
            lb_list=torch.cat([lb_list,labels.view(-1).cpu()])         
            
            if i == 0: pred_score_list = predict_proba.detach().cpu().numpy()
            else:pred_score_list=np.concatenate([pred_score_list,
                                                 predict_proba.detach().cpu().numpy()],axis=0)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss = loss_function(outputs, labels).item()       
    
    accuracy.append(100 * correct/total)
    f1 = f1_score(lb_list,pred_list,average='weighted')   
    roc_auc = auroc(torch.tensor(pred_score_list,dtype = torch.float32), 
                lb_list, num_classes=2)
    bacc = balanced_accuracy_score(pred_list, lb_list)
    spec = specificity(pred_list, lb_list, average='weighted',num_classes=2)
    pre,rec = precision_recall(pred_list, lb_list, average='weighted',num_classes=2)
    
    
    pre_vec, rec_vec, thresholds = precision_recall_curve(torch.tensor(pred_score_list,dtype = torch.float32), lb_list, num_classes=2)
    dlen = len(ds)
    pr_auc= 0
    for idx in range(2):
        w = len((lb_list == idx).nonzero(as_tuple=False))/dlen
        pr_auc = pr_auc+ w*auc(rec_vec[idx], pre_vec[idx])
    
    # Build confusion matrix
    cf_matrix=confusion_matrix(lb_list.numpy(), pred_list.numpy())
            
    df_cm = pd.DataFrame(cf_matrix.astype('int'), index = [i for i in classes],
                          columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True,fmt="g")
    con_path = 'confusion_matrix_test/'+model_name
    os.makedirs(con_path, exist_ok=True)
    plt.savefig(con_path+'/output.png')                        
    
    
      
    # display the test results   
    print('Acc: %.2f, Balenced Acc.:%.2f, spec.:%.4f, pre.:%.4f, rec.:%.4f, F1: %.4f, ROC AUC: %.4f, PR AUC: %.4f'
          %(100*correct/total,bacc*100,spec,pre,rec, f1, roc_auc, pr_auc ))
    print('Confusion matrix')
    print(metrics.confusion_matrix(lb_list.numpy(), pred_list.numpy()))
    print(metrics.classification_report(lb_list.numpy(), pred_list.numpy(),digits=4))
    print(metric_report(lb_list.numpy(), pred_list.numpy()))
    return 100*correct/total
#%%
seed = 0
set_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
loss_function = nn.CrossEntropyLoss().cuda()

model_path = 'model_trained/proposed'    
model = Model()  
model.load_state_dict(torch.load(model_path+'/trained_model.pt'))
model.cuda() 
data_path = 'Data/classification/'   
model_name = 'proposed'
params = {
    'batch_size': 48,
    'data_path': data_path,
    'loss_function':loss_function,
    'model_name':model_name,
    'device':device
    }        
preds=test(model, params)
torch.cuda.empty_cache()

