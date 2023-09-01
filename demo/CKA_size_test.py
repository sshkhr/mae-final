#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import transforms
from einops import rearrange, repeat
from loguru import logger
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import numpy as np
from cka import gram, centering_mat, centered_gram, unbiased_hsic_xy, MinibatchCKA, HookedCache, make_pairwise_metrics, update_metrics, get_simmat_from_metrics
import pickle
from tqdm import tqdm

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import timm
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')


# In[2]:


from torch.nn.modules.container import Sequential
from torchvision.models.resnet import Bottleneck
from torchvision.models.vision_transformer import EncoderBlock


# In[3]:


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


# # ViT vs ResNet (sanity checks)

# In[4]:


vit_b_32 = torchvision.models.vit_b_32(pretrained=True)
resnet50 = torchvision.models.resnet50(pretrained=True)


# In[5]:


#%%time


t_layers  = ([name for name, module in vit_b_32.named_modules()
              if 'drop' not in name and 'out_proj' not in name and name != '' 
              and not isinstance(module, Sequential) and not isinstance(module, EncoderBlock)])
t_layers = np.random.choice(t_layers, 10)
modt_hooks = [HookedCache(vit_b_32, name) for name in t_layers]

c_layers  = ([name for name, module in resnet50.named_modules() 
              if 'drop' not in name and name != '' and name != ''               
              and not isinstance(module, Sequential) and not isinstance(module, Bottleneck)])

c_layers = np.random.choice(c_layers, 10)
modc_hooks = [HookedCache(vit_b_32, name) for name in t_layers]


# In[6]:


vit_b_32.to(DEVICE)
resnet50.to(DEVICE)

resnet50.eval()
vit_b_32.eval()

# In[7]:


log_every = 10
do_log = True


# ### Sweep over different dataset sizes

# In[8]:

batch_sizes = [32*(i) for i in range(1, 4)] + [128*(i) for i in range(1, 8)]

sim_mat_cc = {i:None for i in batch_sizes}
sim_mat_tt = {i:None for i in batch_sizes}
sim_mat_ct = {i:None for i in batch_sizes}


# In[9]:


for i in tqdm(batch_sizes):

    print('Batch size:', i)

    indexes = np.random.uniform(low=0, high=len(dataset), size=10240).astype(int)
    subset = torch.utils.data.Subset(dataset, indexes)

    data_loader = torch.utils.data.DataLoader(subset,
                                          batch_size=i,
                                          shuffle=False,
                                          pin_memory=False)

    metrics_ct = make_pairwise_metrics(modc_hooks, modt_hooks, device=DEVICE)
    metrics_cc = make_pairwise_metrics(modc_hooks, modc_hooks, device=DEVICE)
    metrics_tt = make_pairwise_metrics(modt_hooks, modt_hooks, device=DEVICE)

    with torch.no_grad():
    
        for it, [batch, label] in enumerate(data_loader):
            batch = batch.to(DEVICE)
            do_log =  (it % log_every == 0)
            if do_log:
               logger.debug(f"iter: {it}")
            outv_c = resnet50(batch)
            outv_t = vit_b_32(batch)

            update_metrics(modc_hooks, modt_hooks, metrics_ct, "cka/ct", do_log,  device=DEVICE)
            update_metrics(modc_hooks, modc_hooks, metrics_cc, "cka/cc", do_log,  device=DEVICE)
            update_metrics(modt_hooks, modt_hooks, metrics_tt, "cka/tt", do_log,  device=DEVICE)
                
            for hook0 in modc_hooks:
                for hook1 in modt_hooks:
                    hook0.clear()
                    hook1.clear()

    sim_mat_cc[i] = get_simmat_from_metrics(metrics_cc)
    sim_mat_tt[i] = get_simmat_from_metrics(metrics_tt)
    sim_mat_ct[i] = get_simmat_from_metrics(metrics_ct)

    

with open('sim_mat_cc_batch.pkl', 'wb') as f:
    pickle.dump(sim_mat_cc, f)

with open('sim_mat_ct_batch.pkl', 'wb') as f:
    pickle.dump(sim_mat_ct, f)

with open('sim_mat_tt_batch.pkl', 'wb') as f:
    pickle.dump(sim_mat_tt, f)




# In[17]:

'''
figure(figsize=(3, 2), dpi=300)

sim_mat = get_simmat_from_metrics(metrics_tt)
ax = sns.heatmap(sim_mat)
ax.invert_yaxis()
plt.title('ViT-B/32-ViT-B/32')
plt.show()

figure(figsize=(3, 2), dpi=300)

sim_mat = get_simmat_from_metrics(metrics_cc)
ax = sns.heatmap(sim_mat)
ax.invert_yaxis()
plt.title('ResNet50-ResNet50')
plt.show()

figure(figsize=(3, 2), dpi=300)

sim_mat = get_simmat_from_metrics(metrics_ct)
ax = sns.heatmap(sim_mat)
ax.invert_yaxis()
plt.title('ViT-B/32-ResNet50')
plt.show()
'''
