#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import os
import pandas as pd
import sys
import requests
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
import seaborn as sns
from matplotlib.pyplot import figure
from cka import gram, centering_mat, centered_gram, unbiased_hsic_xy, MinibatchCKA, HookedCache, make_pairwise_metrics, update_metrics, get_simmat_from_metrics
import numpy as np
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import timm
from torch.utils.tensorboard import SummaryWriter

sys.path.append('..')
import models_mae

writer = SummaryWriter()

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')


# In[ ]:


from torch.nn.modules.container import Sequential
from torchvision.models.resnet import Bottleneck
from torchvision.models.vision_transformer import EncoderBlock
from timm.models.vision_transformer import Block


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


indexes = np.random.uniform(low=0, high=len(dataset), size=1024).astype(int)
subset = torch.utils.data.Subset(dataset, indexes)

data_loader = torch.utils.data.DataLoader(subset,
                                        batch_size=128,
                                        shuffle=False,
                                        pin_memory=False)

log_every = 10
do_log = True


# # ViT vs ResNet (sanity checks)

# In[ ]:


vit_b_32 = torchvision.models.vit_b_32(pretrained=True)
vit_l_16 = torchvision.models.vit_l_16(pretrained=True)
resnet50 = torchvision.models.resnet50(pretrained=True)


# In[ ]:


#%%time

log_every = 10

modt_hooks = []
for name, module in vit_b_32.named_modules():
  if 'drop' not in name and 'out_proj' not in name and name != '' and not isinstance(module, Sequential) and not isinstance(module, EncoderBlock):
    print(name, type(module))
    tgt = name
    hook = HookedCache(vit_b_32, tgt)
    modt_hooks.append(hook)

modtl_hooks = []
for name, module in vit_l_16.named_modules():
  if 'drop' not in name and 'out_proj' not in name and name != '' and not isinstance(module, Sequential) and not isinstance(module, EncoderBlock):
    print(name, type(module))
    tgt = name
    hook = HookedCache(vit_l_16, tgt)
    modtl_hooks.append(hook)

modc_hooks = []
for name, module in resnet50.named_modules():
  if 'drop' not in name and name != '' and name != '' and not isinstance(module, Sequential) and not isinstance(module, Bottleneck):
    print(name, type(module))
    tgt = name
    hook = HookedCache(resnet50, tgt)
    modc_hooks.append(hook)


# In[ ]:


#vit_b_32.to(DEVICE)
vit_l_16.to(DEVICE)
resnet50.to(DEVICE)

#vit_b_32.eval()
vit_l_16.eval()
resnet50.eval()


# In[ ]:


#metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks, device=DEVICE)
#metrics_cc = make_pairwise_metrics(modc_hooks, modc_hooks, device=DEVICE)
#metrics_tt = make_pairwise_metrics(modt_hooks, modt_hooks, device=DEVICE)
metrics_tltl = make_pairwise_metrics(modtl_hooks, modtl_hooks, device=DEVICE)
metrics_ctl = make_pairwise_metrics(modc_hooks, modtl_hooks, device=DEVICE)


with torch.no_grad():
  
  for it, [batch, label] in enumerate(data_loader):
    batch = batch.to(DEVICE)
    do_log =  (it % log_every == 0)
    if do_log:
      logger.debug(f"iter: {it}")
    outv_c = resnet50(batch)
    outv_t = vit_l_16(batch)

    #update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", do_log,  device=DEVICE)
    #update_metrics(modc_hooks, modc_hooks, metrics_cc, "cka/cc", do_log,  device=DEVICE)
    #update_metrics(modt_hooks, modt_hooks, metrics_tt, "cka/tt", do_log,  device=DEVICE)
    update_metrics(modtl_hooks, modtl_hooks, metrics_tltl, "cka/tltl", do_log,  device=DEVICE)
    update_metrics(modc_hooks, modtl_hooks, metrics_ctl, "cka/tt", do_log,  device=DEVICE)
        
    for hook0 in modc_hooks:
      for hook1 in modtl_hooks:
        hook0.clear()
        hook1.clear()


# In[ ]:


#sim_mat_cc = get_simmat_from_metrics(metrics_cc)
#sim_mat_ct = get_simmat_from_metrics(metrics_ct)
#sim_mat_tt = get_simmat_from_metrics(metrics_tt)
sim_mat_tltl = get_simmat_from_metrics(metrics_tltl)
sim_mat_ctl = get_simmat_from_metrics(metrics_ctl)


# In[ ]:

'''
with open('sim_mat_resnet50.pkl', 'wb') as f:
    pickle.dump(sim_mat_cc, f)

with open('sim_mat_resnet50_vit-b-32.pkl', 'wb') as f:
    pickle.dump(sim_mat_ct, f)

with open('sim_mat_vit-b-32.pkl', 'wb') as f:
    pickle.dump(sim_mat_tt, f)
'''

with open('vit-l-16.pkl', 'wb') as f:
    pickle.dump(sim_mat_tltl, f)

with open('sim_mat_resnet50_vit-l-16.pkl', 'wb') as f:
    pickle.dump(sim_mat_ctl, f)


# In[ ]:


figure(figsize=(3, 2), dpi=300)

ax = sns.heatmap(sim_mat_tt)
ax.invert_yaxis()
plt.title('ViT-B/32-ViT-B/32')
plt.show()

figure(figsize=(3, 2), dpi=300)

ax = sns.heatmap(sim_mat_cc)
ax.invert_yaxis()
plt.title('ResNet50-ResNet50')
plt.show()

figure(figsize=(3, 2), dpi=300)

ax = sns.heatmap(sim_mat_ct)
ax.invert_yaxis()
plt.title('ViT-B/32-ResNet50')
plt.show()

