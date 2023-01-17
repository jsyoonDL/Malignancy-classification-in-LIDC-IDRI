- Hi, I’m @jsyoonDL 
- This is for nodule classification using LIDC-IDRI dataset.
- Target : classification the nodule is malignancy or not.
- The dataset can download the below link.

   -> Dataset (LIDC-IDRI) : https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254

- Contact

   -> giseg2118@gmail.com & jsyoon@tukorea.ac.kr

The code is based on 


- torch == '2.0.0' 
- timm == '0.6.12'
- pylidc == '0.2.3'
- torchmetrics == '0.9.3'
- sklean == '1.0.2'
- kornia == '0.6.8'


<**ROC analysis**>

Acc: 81.09, Balenced Acc.:75.21, spec.:0.8109, pre.:0.7964, rec.:0.8109, F1: 0.7957 
ROC AUC: 0.7671, PR AUC: 0.8106

<**Confusion matrix**>
![output](https://user-images.githubusercontent.com/87634620/212791219-1bcc7054-6ff5-4787-bf4b-51a3b0c50f03.png){:width="50%" height="50%"}

   
<**ROC analysis#1**>   
|                |precision|recall   |f1-score|  support|
|----------------|---------|---------|--------|---------|
|**0**           |   0.8376|   0.9314|  0.8820|      598|
|**1**           |   0.6667|   0.4316|  0.5240|      190|
|**accuracy**    |         |         |  0.8109|      788|
|**macro avg**   |   0.7521|   0.6815|  0.7030|      788|
|**weighted avg**|   0.7964|   0.8109|  0.7957|      788|

<**ROC analysis#2**> 
|     |Sensitivity|Specificity|Precision|      ACC|
|-----|-----------|-----------|---------|---------|
|**0**|   0.931438|   0.431579| 0.837594| 0.810914|
|**1**|   0.431579|   0.931438| 0.666667| 0.810914|
