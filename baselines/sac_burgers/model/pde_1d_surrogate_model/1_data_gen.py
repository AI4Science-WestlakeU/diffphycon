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
    parser.add_argument('--num_f', default=10, type=int, help='the number of force data')
    parser.add_argument('--num_u0', default=10, type=int, help='the number of initial data')
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
    
    #make data
    u0, f = make_data(Nu0=args.num_u0, Nf=args.num_f, s=args.grid_size)
    
    ##Solve
    u0=torch.FloatTensor(u0)
    f=torch.FloatTensor(f)
    print("Make trainset to learn solution operator: number of force={}, number of initial state={}".format(args.num_f, args.num_u0))
    trajectory = burgers_numeric_solve(u0, f, visc=0.01, T=1.0, dt=1e-4, num_t=args.num_t, mode='const')
    
    torch.save([f.cpu(), trajectory.cpu()], 'data/burgers_dataset')