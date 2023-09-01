#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from genericpath import isfile
from pathlib import Path
import os
import pandas as pd
import sys
import requests
from functools import partial
from tqdm import tqdm

import torch
import torch.distributed as dist
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
import numpy as np
import pandas as pd
import re
import submitit
import fnmatch

from torchsummary import summary
from sklearn.manifold import TSNE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import timm
from timm.utils import accuracy

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
  DEVICE = torch.device('cuda')

from get_models import get_model
from knn_probe import extract_features

import seaborn as sns

# ### Create ImageNet validation dataset

# In[ ]:


class ReturnIndexDataset(torchvision.datasets.ImageFolder):
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

data_path = '/datasets01/imagenet_full_size/061417/'

dataset_train = ReturnIndexDataset(os.path.join(data_path, "train"), transform=transform)
dataset_val = ReturnIndexDataset(os.path.join(data_path, "val"), transform=transform)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=512,
    num_workers=10,
    pin_memory=True,
    drop_last=False,
)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=512,
    num_workers=10,
    pin_memory=True,
    drop_last=False,
)
print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")


def get_features(name, mode):

    folder_name = os.path.join('../features', name, mode)
    print(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    try:
        train_features = torch.load(folder_name + "/" + "slurm_trainfeat.pth")
        with open(folder_name + "/" + "slurm_traincache.pkl", "rb") as file:
            train_cache = pickle.load(file)
    except OSError or EOFError:
        # need to extract features !
        train_features, train_cache  = extract_features(model, data_loader_train)
        train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
        for agg in train_cache:
            for layer in train_cache[agg]:
                train_cache[agg][layer] = torch.nn.functional.normalize(train_cache[agg][layer], dim=1, p=2)
        torch.save(train_features.cpu(), folder_name + "/" + "slurm_trainfeat.pth")
        pickle.dump(train_cache, open(folder_name + "/" + "slurm_traincache.pkl", "wb" ))

    try:
        test_features = torch.load(folder_name + "/" + "slurm_testfeat.pth")
        with open(folder_name + "/" + "slurm_testcache.pkl", "rb") as file:
            test_cache = pickle.load(file)
    except OSError or EOFError:
        # need to extract features !
        test_features, test_cache  = extract_features(model, data_loader_val)
        test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)
        for agg in test_cache:
            for layer in test_cache[agg]:
                test_cache[agg][layer] = torch.nn.functional.normalize(test_cache[agg][layer], dim=1, p=2)
        torch.save(test_features.cpu(), folder_name + "/" + "slurm_testfeat.pth")
        pickle.dump(test_cache, open(folder_name + "/" + "slurm_testcache.pkl", "wb" ))
        
    print(train_features.shape)
    print(test_features.shape)

    for agg in train_cache:
        for layer in train_cache[agg]:
            print(agg, layer, train_cache[agg][layer].shape)

    for agg in test_cache:
        for layer in test_cache[agg]:
            print(agg, layer, test_cache[agg][layer].shape)

    try:
        train_labels = torch.load(folder_name + "/" + "slurm_trainlabels.pth")
    except OSError:
        train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
        torch.save(train_labels.cpu(), folder_name + "/" + "slurm_trainlabels.pth")

    try:
        test_labels = torch.load(folder_name + "/" + "slurm_testlabels.pth")
    except OSError:
        test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
        torch.save(train_labels.cpu(), folder_name + "/" + "slurm_testlabels.pth")

    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

    return train_features, test_features, train_cache, test_cache, train_labels, test_labels

def plot_and_save(tsne_results, test_labels, name, mode, agg, layer):

    file_name = '../figures/tsne/' + name + '/' + mode + '/' + agg + '_' + layer + '.png'

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    fig = plt.figure(figsize=(16, 16), dpi=300) 
    
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=test_labels,
        palette=sns.color_palette("hls", 1000),
        legend=False,
        alpha=0.3
    )

    plt.title('2D tSNE embedding of ' + mode + ' ' + name + ' ' + layer + ' ' + agg)

    fig.savefig(file_name)


def tsne_plot(name, mode, agg, layer):

    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)

    file_name = 'tsne_results/' + name + '/' + mode + '/' + agg + '_' + layer + "_" + "tsne.pkl"

    print(file_name)

    Path(os.path.dirname(file_name)).mkdir(parents=True, exist_ok=True)

    if os.path.isfile(file_name):
        with open(file_name, 'rb') as pickle_file:
            tsne_results = pickle.load(pickle_file)
        test_labels = [s[-1] for s in dataset_val.samples]
        print("Loaded File:", file_name)
    else:
        train_features, test_features, train_cache, test_cache, train_labels, test_labels = get_features(name, mode)
        test_labels = [s[-1] for s in dataset_val.samples]
        tsne_results = tsne.fit_transform(test_cache[agg][layer].numpy())
        pickle.dump(tsne_results, open(file_name, "wb"))

    plot_and_save(tsne_results, test_labels, name, mode, agg, layer)


if __name__ == "__main__":

    executor = submitit.AutoExecutor(folder='tsne-logs')
    executor.update_parameters(
        mem_gb=256,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        timeout_min=600,
        slurm_partition="learnlab",
        slurm_signal_delay_s=120,
    )

    jobs = []
    names = ['MoCo-V3', 'DINO', 'MAE']
    modes = ['linear', 'finetuned']
    aggs = ['CLS_feats']
    layers = ['blocks.' + str(i) for i in range(12)]

    with executor.batch():
        for name in names:
            for mode in modes:
                for agg in aggs:
                    for layer in layers:
                        print(name, mode, agg, layer)
                        file_name = 'tsne_results/' + name + '/' + mode + '/' + agg + '_' + layer + "_" + "tsne.pkl"
                        if not os.path.isfile(file_name):    
                            job = executor.submit(tsne_plot, name, mode, agg, layer)
                            jobs.append(job)