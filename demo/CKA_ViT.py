#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
from pathlib import Path
import os
import pandas as pd
import sys
import requests
from functools import partial

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
from get_models import get_model
import submitit
from matplotlib.pyplot import figure

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import timm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')
  print("Using CUDA")


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
                                        batch_size=32,
                                        shuffle=False,
                                        pin_memory=False)

log_every = 1
do_log = True


# # Get models

# In[ ]:

def get_eval_model(name, mode):
    print(name, mode)
    model = get_model(name, mode)
    model.to(DEVICE)
    model.eval()

    return model


def get_hooks(model, type='vit'):

    sys.path.append('../')
    sys.path.append('../msn')

    from timm.models.vision_transformer import Block as timm_block
    from vision_transformer import Block
    from torch.nn.modules.container import Sequential
    from torchvision.models.resnet import Bottleneck

    from src.deit import Block as msn_block

    hooks = []
    
    print(model.named_modules(), type)

    if type == 'vit':
        for name, module in model.named_modules():
            if ('drop' not in name and 'ls' not in name and 'out_proj' not in name and 'decoder' not in name and ('block' in name or 'encoder_layer' in name) and 'head' not in name and (name != 'blocks' and name!= 'encoder.layers' and 'encoder' not in name.split('.')[-1]) and not isinstance(module, Block)and not isinstance(module, msn_block) and not isinstance(module, timm_block)):
                tgt = name
                hook = HookedCache(model, tgt)
                hooks.append(hook)
                print(name)
    elif type == 'cnn':
        for name, module in model.named_modules():
            if 'drop' not in name and name != 'downsample_layers' and name != 'stages' and name != 'norm' and name != 'head':
                tgt = name
                hook = HookedCache(model, tgt)
                hooks.append(hook)

    return hooks


@torch.no_grad()
def get_cka(model1_details, model2_details):

    model1_name, model1_mode = model1_details
    model2_name, model2_mode = model2_details
    
    folder = "sim_mat/"
    file_name = folder + 'sim_mat_' + model1_name + '_' + model1_mode + '_' + model2_name + '_' + model2_mode + '.pkl'

    if not os.path.isfile(file_name):
        model1 = get_eval_model(model1_name, model1_mode)
        model2 = get_eval_model(model2_name, model2_mode)

        type1 = 'cnn' if 'Conv' in model1_name else 'vit'
        hooks1 = get_hooks(model1, type=type1)
        type2 = 'cnn' if 'Conv' in model2_name else 'vit'
        hooks2 = get_hooks(model2, type=type2)

        if model2_name == 'SimMIM':
            del hooks2[3::10]

        print(file_name)
        print(len(hooks1))
        print(len(hooks2))

        #input()
        
        metrics = make_pairwise_metrics(hooks1, hooks2, device=DEVICE)

        with torch.no_grad():  
            for it, [batch, label] in enumerate(data_loader):
                batch = batch.to(DEVICE)
                do_log =  (it % 1 == 0)
                if do_log:
                    logger.debug(f"iter: {it}")

                out1 = model1(batch)
                out2 = model2(batch)
                
                print("Done feedforward")

                update_metrics(hooks1, hooks2, metrics, "cka", do_log,  device=DEVICE)

                for hook1 in hooks1:
                    for hook2 in hooks2:
                        hook1.clear()
                        hook2.clear()

        sim_mat = get_simmat_from_metrics(metrics)

        with open(file_name, 'wb') as f:
            pickle.dump(sim_mat, f)
    else:
        with open(file_name, 'rb') as f:
            sim_mat = pickle.load(f)

    save_figures(sim_mat, model1_name, model1_mode, model2_name, model2_mode)


def save_image(sim_mat, fig_dir, name, indexes = None):
    fig = figure(figsize=(9, 6), dpi=500) 

    if indexes is not None:
        sim_mat = sim_mat[indexes, :][:, indexes]

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sns.set(font_scale=1.5)

    ax = sns.heatmap(sim_mat, vmin = 0.0, vmax = 1.0)
    ax.invert_yaxis()
    model1 = name.split(' - ')[0] + ' layers'
    model2 = name.split(' - ')[1] + ' layers'
    plt.ylabel(model1, fontweight="bold")
    plt.xlabel(model2, fontweight="bold")

    if ((not 'mlp' in fig_dir) and (not 'att' in fig_dir) and (not 'norm' in fig_dir)):
        ax.set_xticks(np.arange(0, 120, 10))
        ax.set_yticks(np.arange(0, 120, 10))
        ax.set_xticklabels(str(x) for x in np.arange(0, 120, 10))
        ax.set_yticklabels(str(x) for x in np.arange(0, 120, 10))

    #plt.title(name)
    plt.show()
    fig.savefig(fig_dir + name + '.pdf', dpi=500, bbox_inches = "tight")


def save_figures(sim_mat, model1_name, model1_mode, model2_name, model2_mode):

    fig_dir = '../figures/CKA-pdf-rebuttal/'
    name = model1_name + ' (' + model1_mode + ') - ' + model2_name + ' (' + model2_mode + ')'

    save_image(sim_mat, fig_dir, name)

    # Layer-wise

    layer_split = {'normalization': list(range(0, 108, 9)) + list(range(4, 108 ,9)),
                    'attention': list(range(1, 108, 9)) + list(range(2, 108 ,9)) + list(range(3, 108 ,9)),
                    'mlp_layers': list(range(5, 108, 9)) + list(range(6, 108 ,9)) + list(range(7, 108 ,9)) + list(range(8, 108 ,9)),
                    'norm1': list(range(0, 108, 9)),
                    'attn':  list(range(1, 108, 9)),
                    'attn_qkv': list(range(2, 108, 9)),
                    'attn_proj': list(range(3, 108, 9)),
                    'norm2': list(range(4, 108, 9)),
                    'mlp': list(range(5, 108, 9)),
                    'mlp_fc1': list(range(6, 108, 9)),
                    'mlp_act': list(range(7, 108, 9)),
                    'mlp_fc2': list(range(8, 108, 9)),
                    }

    for layer in layer_split:
        indexes = layer_split[layer]
        indexes.sort()
        
        fig_dir = '../figures/CKA-pdf-rebuttal/' + layer + '/'
        save_image(sim_mat, fig_dir, name, indexes)

if __name__ == "__main__":

    executor = submitit.AutoExecutor(folder='CKA-logs-rebuttal')
    executor.update_parameters(
        mem_gb=64,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        timeout_min=120,
        slurm_partition="devlab",
        slurm_signal_delay_s=120,
    )
    
    jobs = []
    names = ['DINO', 'MAE', 'MoCo-V3']
    modes = ['finetuned', 'pretrained']
    prod = list(itertools.product(names, modes))

    #conv = [('ConvNext', 'pretrained'), ('ConvNext', 'finetuned')]
    #conv = ('SiameseIM', 'pretrained'), ('SiameseIM', 'finetuned')]
    #conv = [('SimMIM', 'pretrained'), ('SimMIM', 'finetuned')]
    conv = [('Supervised', 'finetuned')]

    #get_cka(prod[0], conv[0])
    
    
    with executor.batch():
        for comb in itertools.product(prod, conv):
        #for comb in itertools.combinations(prod, 2):
            print(comb)
            job = executor.submit(get_cka, *comb)
            jobs.append(job)
    