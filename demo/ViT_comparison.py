#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import os
import pandas as pd
import sys
import requests
from functools import partial
from tqdm import tqdm

import torch
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch 
from einops import rearrange, repeat
from loguru import logger
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import seaborn as sns
from matplotlib.pyplot import figure
from cka import gram, centering_mat, centered_gram, unbiased_hsic_xy, MinibatchCKA, HookedCache, make_pairwise_metrics, update_metrics, get_simmat_from_metrics
import numpy as np
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import timm
from timm.utils import accuracy
from torch.utils.tensorboard import SummaryWriter

from get_models import get_model

writer = SummaryWriter()

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')


# ### Create ImageNet validation dataset

# In[ ]:


# using the validation transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

## Randomly subsample features too

data_path='/datasets01/imagenet_full_size/061417/'
dataset = torchvision.datasets.ImageFolder(data_path + 'val', transform=transform)


# In[ ]:


data_loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=128,
                                        shuffle=False,
                                        pin_memory=False)

log_every = 10
do_log = True


# ### Load pre-trained DINO, Moco-v3, MAE with linear classifiers

# In[ ]:


vitb16_dino = get_model('DINO', 'linear')
vitb16_mae = get_model('MAE', 'linear')
vitb16_mocov3 = get_model('MoCo-V3', 'linear')
vitb16_dino_ft = get_model('DINO', 'finetuned')
vitb16_mae_ft = get_model('MAE', 'finetuned')
vitb16_mocov3_ft = get_model('MoCo-V3', 'finetuned')
vitb16_supervised = get_model('Supervised')


# In[ ]:


vitb16_dino.to(DEVICE)
vitb16_mae.to(DEVICE)
vitb16_mocov3.to(DEVICE)
vitb16_dino_ft.to(DEVICE)
vitb16_mae_ft.to(DEVICE)
vitb16_mocov3_ft.to(DEVICE)
vitb16_supervised.to(DEVICE)

vitb16_supervised.eval()
vitb16_dino.eval()
vitb16_mae.eval()
vitb16_mocov3.eval()
vitb16_dino_ft.eval()
vitb16_mae_ft.eval()
vitb16_mocov3_ft.eval()


# ### Validate on ImageNet test set

# In[ ]:


def validate(dataloader, model, device):

    results = torch.zeros(len(dataloader.dataset), 2)
    rankings = torch.zeros(len(dataloader.dataset), 1000)

    with torch.no_grad():
        
        # use with one image per batch
        for it, [batch, target] in enumerate(tqdm(dataloader)):
            batch = batch.to(device)
            target = target.to(device)
            output = model(batch)
            
            batch_size = target.size(0)

            
            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
            results[it*batch_size:(it+1)*batch_size, 0] = correct[:1].view(-1).float() 
            results[it*batch_size:(it+1)*batch_size, 1] = correct[:5].sum(0).view(-1).float()
            

            _, rankings[it*batch_size:(it+1)*batch_size, :] = output.topk(1000, 1, True, True)

    return results, rankings


# In[ ]:


moco_results, moco_ranks = validate(data_loader, vitb16_mocov3, DEVICE)


# In[ ]:


dino_results, dino_ranks = validate(data_loader, vitb16_dino, DEVICE)


# In[ ]:


mae_results, mae_ranks = validate(data_loader, vitb16_mae, DEVICE)


# In[ ]:


moco_results_ft, moco_ranks_ft = validate(data_loader, vitb16_mocov3_ft, DEVICE)


# In[ ]:


dino_results_ft, dino_ranks_ft = validate(data_loader, vitb16_dino_ft, DEVICE)


# In[ ]:


mae_results_ft, mae_ranks_ft = validate(data_loader, vitb16_mae_ft, DEVICE)


# In[ ]:


supervised_results, supervised_ranks = validate(data_loader, vitb16_supervised, DEVICE)


# In[ ]:


dic = {'supervised': [supervised_results, supervised_ranks],
       'MAE': [mae_results, mae_ranks],
       'DINO': [dino_results, dino_ranks],
       'MoCo-V3': [moco_results, moco_ranks],
       'MAE-FT': [mae_results_ft, mae_ranks_ft],
       'DINO-FT': [dino_results_ft, dino_ranks_ft],
       'MoCo-V3-FT': [moco_results_ft, moco_ranks_ft],
        }


# In[ ]:


import pickle

pickle.dump(dic, open("ranks_and_results.pkl", "wb" ))


