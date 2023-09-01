#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import os
import pandas as pd
import sys
import requests
from functools import partial, wraps
from tqdm import tqdm
import json

import torch
import torch.distributed as dist
import torchvision
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch 
from einops import rearrange, repeat
from loguru import logger
from torchmetrics import Metric, BootStrapper
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import re
import submitit
import fnmatch
import random

from torchsummary import summary

from torchvision.datasets import ImageNet

def file_cache(filename):
    """Decorator to cache the output of a function to disk."""
    def decorator(f):
        @wraps(f)
        def decorated(self, directory, *args, **kwargs):
            filepath = Path(directory) / filename
            if filepath.is_file():
                out = json.loads(filepath.read_text())
            else:
                out = f(self, directory, *args, **kwargs)
                filepath.write_text(json.dumps(out))
            return out
        return decorated
    return decorator

class CachedImageNet(ImageNet):
    @file_cache(filename="/private/home/sshkhr/in1k_cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="/private/home/sshkhr/in1k_cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import timm
from timm.utils import accuracy

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')

from get_models import get_model

# ### Create ImageNet validation dataset

# In[ ]:


class ReturnIndexDataset(torchvision.datasets.ImageFolder):
    @file_cache(filename="/private/home/sshkhr/in1k_cached_classes.json")
    def find_classes(self, directory, *args, **kwargs):
        classes = super().find_classes(directory, *args, **kwargs)
        return classes

    @file_cache(filename="/private/home/sshkhr/in1k_cached_structure.json")
    def make_dataset(self, directory, *args, **kwargs):
        dataset = super().make_dataset(directory, *args, **kwargs)
        return dataset

    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


# In[ ]:


transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

transform_flip = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

data_path = '/datasets01/imagenet_full_size/061417/'

print("Loading dataset...")
dataset_train = ReturnIndexDataset(os.path.join(data_path, "val"), transform=transform)
dataset_train_transformed = ReturnIndexDataset(os.path.join(data_path, "val"), transform=transform_flip)

indices = random.sample(range(1, len(dataset_train)), 10000)

index_keys = {indices[i]:i for i in range(len(indices))}

dataset_train = Subset(dataset_train, indices)
dataset_train_transformed = Subset(dataset_train_transformed, indices)

#dataset_val = ReturnIndexDataset(os.path.join(data_path, "val"), transform=transform)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=512,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
)
data_loader_transformed = torch.utils.data.DataLoader(
    dataset_train_transformed,
    batch_size=512,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
)
print(f"Data loaded with {len(dataset_train)} train and {len(dataset_train_transformed)} val imgs.")


# ### Load pre-trained DINO, Moco-v3, MAE

# In[ ]:

def getActivation(name, cache):

    cache.setdefault('CLS_feats', {})
    #cache.setdefault('GAP_feats', {})
    #cache.setdefault('GAP_WO_CLS_feats', {})

    # the hook signature
    def hook(model, input, output):
            cls_features = output[:, 0].detach().cpu()
            gap_features = output.mean(dim=-2).detach().cpu()
            gap_features_wo_cls = output[:, 1:].mean(dim=-2).detach().cpu()
            
            cache['CLS_feats'].setdefault(name, []).append(cls_features)
            #cache['GAP_feats'].setdefault(name, []).append(gap_features)
            #cache['GAP_WO_CLS_feats'].setdefault(name, []).append(gap_features_wo_cls)
    return hook

# In[ ]:


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    sys.path.append('../dino')
    import utils
    
    metric_logger = utils.MetricLogger(delimiter="  ")

    features = None

    cache = {'CLS_feats':{}}#, 'GAP_feats:':{}, 'GAP_WO_CLS_feats':{}}
    hooks = []
    intermediate_features = {}

    if use_cuda:
        model = model.cuda()
    
    model.eval()

    indexes = []

    for name, module in model.named_modules():
        if name.startswith('pretrained.'):
            name = name.removeprefix('pretrained.')
        #print(name)
        if fnmatch.fnmatch(name, 'blocks.*') and not fnmatch.fnmatch(name, 'blocks.*.*'):
            hook = module.register_forward_hook(getActivation(name, cache))
            hooks.append(hook)

    for samples, index in tqdm(metric_logger.log_every(data_loader, 10)):
        #print(index)
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        temp = model(samples)
        feats = temp.clone()

        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])

        #if use_cuda:
        #    features = features.cuda()

        new_indexes = torch.LongTensor([index_keys[idx.item()] for idx in index.cpu()])

        if use_cuda:
            features.index_copy_(0, new_indexes, feats.cpu())
        else:
            features.index_copy_(0, new_indexes, feats.cpu())

        indexes.append(new_indexes)


    for hook in hooks:
        hook.remove()
    
    for agg in cache:
        intermediate_features[agg] = {}
        for layer in cache[agg]:
            intermediate_features[agg][layer] = torch.zeros(len(data_loader.dataset), cache[agg][layer][0].shape[-1])

            for index, batch_feat in zip(indexes, cache[agg][layer]):
                intermediate_features[agg][layer].index_copy_(0, index, batch_feat)
            
    return features, intermediate_features


# In[ ]:

def cos_sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

@torch.no_grad()
def invariance(train_features, transformed_features):
    similarity = cos_sim_matrix(transformed_features, train_features)
    shuffled = similarity[torch.randperm(similarity.size()[0])]
    baseline = torch.trace(shuffled)/similarity.shape[0]
    invariance = torch.trace(similarity)/similarity.shape[0]
    normalized_invariance = (baseline -  invariance)/baseline
    return invariance


# ### Validate on ImageNet test set

# In[ ]:


def invariance_probe(name, mode):
    model = get_model(name, mode)

    folder_name = os.path.join('../invariance_features/IN1K-val/', name, mode)
    print(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    try:
        train_features = torch.load(folder_name + "/" + "invariance_trainfeat.pth")
        with open(folder_name + "/" + "invariance_traincache.pkl", "rb") as file:
            train_cache = pickle.load(file)
    except OSError or EOFError:
        # need to extract features !
        train_features, train_cache  = extract_features(model, data_loader_train)
        train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
        for agg in train_cache:
            for layer in train_cache[agg]:
                train_cache[agg][layer] = torch.nn.functional.normalize(train_cache[agg][layer], dim=1, p=2)
        torch.save(train_features.cpu(), folder_name + "/" + "invariance_trainfeat.pth")
        pickle.dump(train_cache, open(folder_name + "/" + "invariance_traincache.pkl", "wb" ))

    try:
        transformed_features = torch.load(folder_name + "/" + "invariance_transformedfeat.pth")
        with open(folder_name + "/" + "invariance_transformedcache.pkl", "rb") as file:
            transformed_cache = pickle.load(file)
    except OSError or EOFError:
        # need to extract features !
        transformed_features, transformed_cache  = extract_features(model, data_loader_transformed)
        transformed_features = torch.nn.functional.normalize(transformed_features, dim=1, p=2)
        for agg in transformed_cache:
            for layer in transformed_cache[agg]:
                transformed_cache[agg][layer] = torch.nn.functional.normalize(transformed_cache[agg][layer], dim=1, p=2)
        torch.save(transformed_features.cpu(), folder_name + "/" + "invariance_transformedfeat.pth")
        pickle.dump(transformed_cache, open(folder_name + "/" + "invariance_transformedcache.pkl", "wb" ))
        
    print(train_features.shape)
    print(transformed_features.shape)

    for agg in train_cache:
        for layer in train_cache[agg]:
            print(agg, layer, train_cache[agg][layer].shape)

    for agg in transformed_cache:
        for layer in transformed_cache[agg]:
            print(agg, layer, transformed_cache[agg][layer].shape)

    print(train_features.shape, transformed_features.shape)

    results = []
    print("Features are ready!\nStart the invariance calculation.")
        
    for agg in transformed_cache:
        for layer in transformed_cache[agg]:
            invariance_score = invariance(train_cache[agg][layer].cpu(), transformed_cache[agg][layer].cpu())
            results.append([agg, layer, invariance_score])
            print(name+" "+agg+" "+layer+f" Invariance score: {invariance_score}")

    invariance_score = invariance(train_features.cpu(), transformed_features.cpu())
    results.append(['linear', 'classifier', invariance_score])
    print("Probe "+f" Invariance score: {invariance_score}")

    df = pd.DataFrame(results, columns=['agg_mode', 'ViT_block', 'invariance_score'])
    df.to_csv(folder_name + "/" + "invariance_results.csv")

    return


# In[ ]:

if __name__ == "__main__":

    '''
    executor = submitit.AutoExecutor(folder='invariance-logs')
    executor.update_parameters(
        mem_gb=512,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        timeout_min=360,
        invariance_partition="devlab,learnfair",
        invariance_signal_delay_s=120,
    )

    jobs = []
    names = ['DINO', 'MoCo-V3', 'MAE']
    modes = ['linear']#, 'finetuned']
    with executor.batch():
        for name in names:
            for mode in modes:
                print(name, mode)
                job = executor.submit(invariance_probe, name, mode)
                jobs.append(job)
    '''

    invariance_probe('DINO', 'linear')
