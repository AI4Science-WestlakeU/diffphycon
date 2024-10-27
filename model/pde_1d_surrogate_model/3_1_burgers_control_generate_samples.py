import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os

from scripts.utils import *
from scripts.burgers_numeric import *

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('--grid_size', default=128, type=int, help = 'grid size')
    parser.add_argument('--num_data', default=100, type=int, help='the number of data')
    parser.add_argument('--num_t', default=10, type=int, help='the number of time interval')
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    return parser.parse_args(argv)


if __name__=='__main__':
    #argparser
    args = get_args()
    print(args)
    
    #gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    use_cuda = torch.cuda.is_available()
    print("Is available to use cuda? : ",use_cuda)
    if use_cuda:
        print("-> GPU number {}".format(args.gpu))
    
    data_all=[]
    f_all=[]
    for i in range(args.num_data):
        #make data
        u0, f = make_data(Nu0=1,Nf=1,s=args.grid_size)
            
        #solve
        u0=torch.FloatTensor(u0)
        f=torch.FloatTensor(f)
        sol = burgers_numeric_solve(u0, f, visc=0.01, T=1.0, dt=1e-4, num_t=args.num_t, mode='const')
        data_all.append(sol)
        f_all.append(f)
    #save data
    torch.save([torch.stack(data_all,dim=0), torch.stack(f_all,dim=0)], 'data/burgers_control_samples')