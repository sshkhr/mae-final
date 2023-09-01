import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch 
from einops import rearrange, repeat
from loguru import logger
from torchmetrics import Metric

class HookedCache:
    def __init__(self, model, target):
        self.model = model
        self.target = target
        
        self.clear()
        self._extract_target()
        self._register_hook()

    @property
    def value(self):
        if isinstance(self._cache, tuple):
            return self._cache[0]
        return self._cache

    def clear(self):
        self._cache = None

    def _extract_target(self):
        for name, module in self.model.named_modules():
          if name == self.target:
              self._target = module
              return

    def _register_hook(self):
        def _hook(module, in_val, out_val):
             self._cache = out_val
        self._target.register_forward_hook(_hook)


def get_simmat_from_metrics(metrics):
    vals = []
    for i, ckas in enumerate(metrics):
      for j, cka in enumerate(ckas):
        z = cka.compute().item()
        vals.append((i,j,z))

    sim_mat = torch.zeros(i+1,j+1)
    for i,j,z in vals:
      sim_mat[i,j] = z
    
    return sim_mat

def make_pairwise_metrics(mod1_hooks, mod2_hooks, device):
  metrics = []
  for i_ in mod1_hooks:
    metrics.append([])
    for j_ in mod2_hooks:
      metrics[-1].append(MinibatchCKA().to(device))
  return metrics

def update_metrics(mod1_hooks, mod2_hooks, metrics, metric_name, do_log, device=None):
    for i, hook1 in enumerate(mod1_hooks):
      for j, hook2 in enumerate(mod2_hooks):
        cka = metrics[i][j]
        X,Y = hook1.value, hook2.value
        
        try:
            cka.update(X,Y, device)
        except AttributeError:
            print('Error')
            print(hook1.target)
            print(hook2.target)
            
          
        if do_log and 0 in (i,j):
          _metric_name = f"{metric_name}_{i}-{j}"
          v = cka.compute()
          
    if do_log:
       sim_mat = get_simmat_from_metrics(metrics)
       sim_mat = sim_mat.unsqueeze(0) * 255
       #writer.add_image(metric_name, sim_mat, it)

def gram(X, device=None):
    if device is not None:
        X.to(device=device)
    # ensure correct input shape
    X = rearrange(X, 'b ... -> b (...)')
    return (X @ X.T).to(device=device)

def centering_mat(n, device=None):
    eye = torch.eye(n)
    ones = torch.ones(n, 1)
    
    if device is not None: 
        ones.to(device=device)
        eye.to(device=device)
    
    H = eye - ((ones @ ones.T) / n)

    return H.to(device=device)

def centered_gram(X, device=None):
    K = gram(X, device=device)
    K = K.float()
    m = K.shape[0]
    H = centering_mat(m, device=device)
    return H @ K @ H

def unbiased_hsic_xy(X, Y, device=None):
    n = X.shape[0]
    assert n > 3 
    ones = torch.ones(n, 1, device=device)
    K = centered_gram(X, device)
    L = centered_gram(Y, device)

    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    
    KL = K @ L
    onesK = ones.T @ K
    Lones = L @ ones
    onesKones = onesK @ ones
    onesLones = ones.T @ Lones

    a = torch.trace(KL)
    b = (onesKones * onesLones) / ((n-1)*(n-2))
    c = (onesK @ Lones) * (2 / (n-2))

    outv = (a + b - c) / (n*(n-3))
    return outv.long().item()

class MinibatchCKA(Metric):
    def __init__(self, dist_sync_on_step=False):
        """
        Introduced in: https://arxiv.org/pdf/2010.15327.pdf
        Implemented to reproduce the results in: https://arxiv.org/pdf/2108.08810v1.pdf
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("_xx", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_xy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_yy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, X: torch.Tensor, Y: torch.Tensor, device=None):
        # NB: torchmetrics Bootstrap resampling janks up batch shape by varying number of samples per batch
        self._xx += unbiased_hsic_xy(X, X, device)
        self._yy += unbiased_hsic_xy(Y, Y, device)
        self._xy += unbiased_hsic_xy(X, Y, device)
        self._batches += 1

    def compute(self):
        xx, xy, yy = self._xx, self._xy, self._yy
        return xy / (torch.sqrt(xx) * torch.sqrt(yy))