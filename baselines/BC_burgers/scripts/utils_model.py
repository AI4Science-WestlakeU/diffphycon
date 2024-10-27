import torch
import numpy as np
import os, sys
from .burgers_numeric import *

def make_data(Nu0, Nf, s):
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax-xmin)/(s+1)
    x = torch.linspace(xmin+delta_x,xmax-delta_x,s)
    
    #make u0
    loc1 = np.random.uniform(0.2, 0.4, (Nu0,1))
    amp1 = np.random.uniform(0, 2, (Nu0,1))
    sig1 = np.random.uniform(0.05, 0.15, (Nu0,1))
    gauss1=amp1*np.exp(-0.5*(np.array(x.view(1,-1).repeat(Nu0,1))-loc1)**2/sig1**2)

    loc2 = np.random.uniform(0.6, 0.8, (Nu0,1))
    amp2 = np.random.uniform(-2, 0, (Nu0,1))
    sig2 = np.random.uniform(0.05, 0.15, (Nu0,1))
    gauss2=amp2*np.exp(-0.5*(np.array(x.view(1,-1).repeat(Nu0,1))-loc2)**2/sig2**2)

    u0=gauss1+gauss2
    
    #make f
    def rand_f(is_rand_amp=True):
        loc = np.random.uniform(0, 1, (Nf,1))
        if is_rand_amp:
            amp = np.random.randint(2, size=(Nf,1)) * np.random.uniform(-1.5, 1.5, (Nf,1))
        else:
            amp = np.random.uniform(-1.5, 1.5, (Nf,1))
        sig = np.random.uniform(0.1, 0.4, (Nf,1))*0.5
        return amp*np.exp(-0.5*(np.array(x.view(1,-1).repeat(Nf,1))-loc)**2/sig**2)
    sum_num_f=7
    f=rand_f(is_rand_amp=False)
    for i in range(sum_num_f):
        f+=rand_f(is_rand_amp=True)

    #solve
    return u0, f

def objective_value(u_T, f, u_star, dt, dx, alpha=0.01):
    u_T, f, u_star = torch.FloatTensor(u_T), torch.FloatTensor(f), torch.FloatTensor(u_star)
    val_u=(1/2)*torch.sum((u_T-u_star)**2)*dx
    val_f=(alpha/2)*torch.sum(f**2)*dx*dt
    return val_u, val_f, val_u+val_f


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
def rel_error2(x, _x):
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
    
    return torch.norm(x - _x, 2, dim=1)

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

