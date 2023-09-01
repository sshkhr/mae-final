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


# ### Load pre-trained DINO, Moco-v3, MAE

# In[ ]:

def getActivation(name, cache):

    cache.setdefault('CLS_feats', {})
    cache.setdefault('GAP_feats', {})
    cache.setdefault('GAP_WO_CLS_feats', {})

    # the hook signature
    def hook(model, input, output):
            cls_features = output[:, 0].detach().cpu()
            gap_features = output.mean(dim=-2).detach().cpu()
            gap_features_wo_cls = output[:, 1:].mean(dim=-2).detach().cpu()
            
            cache['CLS_feats'].setdefault(name, []).append(cls_features)
            cache['GAP_feats'].setdefault(name, []).append(gap_features)
            cache['GAP_WO_CLS_feats'].setdefault(name, []).append(gap_features_wo_cls)
    return hook

# In[ ]:


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    sys.path.append('../dino')
    import utils
    
    metric_logger = utils.MetricLogger(delimiter="  ")

    features = None

    cache = {'CLS_feats':{}, 'GAP_feats:':{}, 'GAP_WO_CLS_feats':{}}
    hooks = []
    intermediate_features = {}

    if use_cuda:
        model = model.cuda()
    
    model.eval()

    indexes = []

    for name, module in model.named_modules():
        if name.startswith('pretrained.'):
            name = name.removeprefix('pretrained.')
        print(name)
        if fnmatch.fnmatch(name, 'blocks.*') and not fnmatch.fnmatch(name, 'blocks.*.*'):
            hook = module.register_forward_hook(getActivation(name, cache))
            hooks.append(hook)

    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        temp = model(samples)
        feats = temp.clone()

        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])

        #if use_cuda:
        #    features = features.cuda()

        if use_cuda:
            features.index_copy_(0, index.cpu(), feats.cpu())
        else:
            features.index_copy_(0, index.cpu(), feats.cpu())

        indexes.append(index.cpu())


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


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


# ### Validate on ImageNet test set

# In[ ]:


def kNN_probe(name, mode):
    model = get_model(name, mode)

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
        torch.save(test_labels.cpu(), folder_name + "/" + "slurm_testlabels.pth")

    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

    results = []
    print("Features are ready!\nStart the k-NN classification.")
    k = 20
        
    for agg in test_cache:
        for layer in test_cache[agg]:
            top1, top5 = knn_classifier(train_cache[agg][layer].cpu(), train_labels.cpu(),
                test_cache[agg][layer].cpu(), test_labels.cpu(), k, 0.07)
            results.append([agg, layer, top1, top5])
            print(name+" "+agg+" "+layer+f" {k}-NN classifier result: Top1: {top1}, Top5: {top5}")

    top1, top5 = knn_classifier(train_features.cpu(), train_labels.cpu(),
            test_features.cpu(), test_labels.cpu(), k, 0.07)
    results.append(['linear', 'classifier', top1, top5])
    print("Probe "+f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")

    df = pd.DataFrame(results, columns=['agg_mode', 'ViT_block', 'top-1', 'top-5'])
    df.to_csv(folder_name + "/" + "knn_results.csv")

    return top1, top5


# In[ ]:

if __name__ == "__main__":

    executor = submitit.AutoExecutor(folder='knn-logs-msn')
    executor.update_parameters(
        mem_gb=512,
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=10,
        timeout_min=360,
        slurm_partition="devlab,learnfair",
        slurm_signal_delay_s=120,
    )

    jobs = []
    names = ['MSN']#['DINO', 'MoCo-V3', 'MAE']
    modes = ['linear']#, 'finetuned']
    with executor.batch():
        for name in names:
            for mode in modes:
                print(name, mode)
                job = executor.submit(kNN_probe, name, mode)
                jobs.append(job)
