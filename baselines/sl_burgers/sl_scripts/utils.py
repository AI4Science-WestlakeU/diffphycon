import torch
import numpy as np
import os, sys
from .burgers_numeric import *

def objective_value(u_T, f, u_star, s, reward_f):
    val_u = np.sum((u_T-u_star)**2) / s
    val_f = np.sum(f**2) / f.shape[0] * 10
    return val_u, val_f, val_u + reward_f * val_f

def error(x, y):
    return torch.norm(x - y)

def rel_error(x, _x):
    """
    <ARGS>
    x : torch.Tensor shape of (B, *)
    _x : torch.Tensor shape of (B, *)
    <RETURN>
    out :torch.Tensor shape of (B), batchwise relative error between x and _x : (||x-_x||_2/||_x||_2)
    
    """
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)
    
    return torch.norm(x - _x, 2, dim=1) / torch.norm(_x, 2, dim=1)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

