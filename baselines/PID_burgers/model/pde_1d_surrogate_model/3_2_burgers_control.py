import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os

from scripts.utils import *
from scripts.burgers_numeric import *
from scripts.burgers_nn import *
from scripts.models import *

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    
    parser.add_argument('name', type=str, help='experiments name')
    parser.add_argument('--data_num', type=int, help='data number')
    parser.add_argument('--grid_size', default=128, type=int, help = 'grid size')
    parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help = 'Number of Epochs')
    parser.add_argument('--lamb1', default=1, type=float, help = 'weight of rec loss')
    parser.add_argument('--lamb2', default=1, type=float, help = 'weight of rec loss')
    parser.add_argument('--gtol', default=1e-5, type=float, help = 'tolerance of gradient')
    parser.add_argument('--ftol', default=1e-5, type=float, help = 'tolerance of function change')
    
    return parser.parse_args(argv)


if __name__=='__main__':
    #argparser
    args = get_args()
    
    #parameter
    lr = args.lr
    epochs = args.epochs
    gtol=args.gtol
    ftol=args.ftol
    # num_interior=128 // num_pts=130=0,dx,...,129*dx // dx=1/129
    s = args.grid_size
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax-xmin)/(s+1)
    x = torch.linspace(xmin+delta_x,xmax-delta_x,s)

    # num_interior=9 // num_pts=11=0,dt,...,10*dt // dt=1/10
    num_t=10 #(= 1/0.1)
    tmin = 0.0; tmax = 1.0
    delta_t = (tmax-tmin)/num_t
    
    num_t_numeric=100 #(= 1/0.01)
    delta_t_numeric = (tmax-tmin)/num_t_numeric
    
    
    #load_data
    u_data, f_data = torch.load('data/burgers_control_samples', map_location=lambda storage, loc: storage)
    
    #load_model
    checkpoint = torch.load('logs/burgers_operator_model', map_location=lambda storage, loc: storage)

    load_model_f = Net_f(s)
    load_model_f.load_state_dict(checkpoint['model_f_state_dict'])
    load_model_f.eval()
    
    load_model_u = Net_u(s)
    load_model_u.load_state_dict(checkpoint['model_u_state_dict'])
    load_model_u.eval()
    
    load_model_trans = Net_trans()
    load_model_trans.load_state_dict(checkpoint['model_trans_state_dict'])
    load_model_trans.eval()
    
    for param in list(load_model_f.parameters())+list(load_model_u.parameters())+list(load_model_trans.parameters()):
        param.requires_grad=False
    

    
    if not os.path.isfile('logs/{}'.format(args.name)):
        logs=dict()
        logs['f_const']=[]
        logs['u_const']=[]

        logs['f_numerical']=[]
        logs['u_numerical']=[]
        logs['t_numerical']=[]
        logs['loss_obj_numerical']=[]
        logs['loss_f_numerical']=[]

        logs['f_nn']=[]
        logs['u_nn']=[]
        logs['t_nn']=[]
        logs['loss_obj_nn']=[]
        logs['loss_f_nn']=[]
    else:
        logs = torch.load('logs/{}'.format(args.name))

    #data
    u_const=u_data[args.data_num].squeeze()
    u_in=u_const[0]
    u_fin=u_const[-1]
    logs['u_const'].append(u_const)

    f_const=f_data[args.data_num].squeeze()
    logs['f_const'].append(f_const)

    print("Do numerical")
    with HiddenPrints():
        f_numerical, t_numerical=burgers_numeric_control(u_in, u_fin, _dt=delta_t_numeric, gtol=gtol, ftol=ftol)
    logs['f_numerical'].append(f_numerical)
    logs['t_numerical'].append(t_numerical)
    with HiddenPrints():
	    u_numerical=burgers_numeric_solve(u_in.unsqueeze(0), f_numerical.unsqueeze(0), visc=0.01, T=1.0, dt=1e-4, num_t=num_t_numeric)
    u_numerical=u_numerical.squeeze()
    logs['u_numerical'].append(u_numerical)
    er_u_numerical, er_f_numerical, er_uf_numerical = objective_value(u_numerical[-1], f_numerical, u_fin, dt=delta_t_numeric, dx=delta_x)
    logs['loss_obj_numerical'].append(er_u_numerical)
    logs['loss_f_numerical'].append(er_f_numerical)

    print("Do neural network")
    f_nn, t_nn=burgers_nn_control(u_in, u_fin, model_f=load_model_f, model_u=load_model_u, model_trans=load_model_trans, epochs=epochs, lr=lr, Nt=num_t, dt=delta_t, dx=delta_x, lamb1=args.lamb1, lamb2=args.lamb2, gtol=gtol, ftol=ftol)
    logs['f_nn'].append(f_nn)
    logs['t_nn'].append(t_nn)
    with HiddenPrints():
        u_nn=burgers_numeric_solve(u_in.unsqueeze(0), f_nn.unsqueeze(0), visc=0.01, T=1.0, dt=1e-4, num_t=num_t)
    u_nn=u_nn.squeeze()
    logs['u_nn'].append(u_nn)
    er_u_nn, er_f_nn, er_uf_nn = objective_value(u_nn[-1], f_nn, u_fin, dt=delta_t, dx=delta_x)
    logs['loss_obj_nn'].append(er_u_nn)
    logs['loss_f_nn'].append(er_f_nn)

    print("Numerical : T={:3.2f}, J={:1.4f}+{:1.4f}={:1.4f}".format(t_numerical, er_u_numerical, er_f_numerical, er_uf_numerical))
    print("Neural    : T={:3.2f}, J={:1.4f}+{:1.4f}={:1.4f}".format(t_nn, er_u_nn, er_f_nn, er_uf_nn))
    
    
    #save
    torch.save(logs, 'logs/{}'.format(args.name))
