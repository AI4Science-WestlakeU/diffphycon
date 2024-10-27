import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import os
import h5py
from datetime import datetime

from scripts.utils import *
from scripts.burgers_numeric import *
from scripts.burgers_nn import *
from scripts.models import *
from model.pde_1d_surrogate_model.burgers_operator import Simu_surrogate_model

import argparse

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    
    parser.add_argument('--reward_f', type=float, default=0, metavar='G', help = 'weight of energy')
    parser.add_argument('--grid_size', default=128, type=int, help = 'grid size')
    parser.add_argument('--lr', default=1e-1, type=float, help ='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help = 'Number of Epochs')
    parser.add_argument('--lamb1', default=1, type=float, help = 'weight of rec loss')
    parser.add_argument('--lamb2', default=1, type=float, help = 'weight of rec loss')
    parser.add_argument('--gtol', default=1e-5, type=float, help = 'tolerance of gradient')
    parser.add_argument('--ftol', default=1e-5, type=float, help = 'tolerance of function change')
    
    return parser.parse_args(argv)


if __name__=='__main__':
    #argparser
    args = get_args()
    print(args)
    
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
    test_dataset = h5py.File("data/free_u_f_1e5/burgers_test.h5", 'r')['test']
    u_data, f_data = torch.tensor(np.array(test_dataset['pde_11-128'])).float()[:50], torch.tensor(np.array(test_dataset['pde_11-128_f'])).float()[:50]
    u_data_full = u_data
    u_data = torch.cat((u_data[:, :, :int(s/4)], u_data[:, :, int(3*s/4):]), -1)
        
    #load_model
    device = torch.device("cuda")
    pde_1d_surrogate_model_checkpoint="checkpoints/partial_ob_full_ctr_1-step"
    milestone=500
    model=Simu_surrogate_model(path=pde_1d_surrogate_model_checkpoint,device=device,s_ob=64,milestone=milestone)
    mse=torch.nn.MSELoss() 
    load_model_f=model.model_f
    load_model_u=model.model_u
    load_model_trans=model.model_trans
    
    for param in list(load_model_f.parameters())+list(load_model_u.parameters())+list(load_model_trans.parameters()):
        param.requires_grad=False
    
    
    # if not os.path.isfile('logs/{}'.format(args.name)):
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
    logs['loss_all_nn']=[]
    logs['state_loss']=[]
    logs['rel_state_loss']=[]
    # else:
    #     logs = torch.load('logs/{}'.format(args.name))
    date = datetime.now()
    u_nn_surrogate_all = []
    u_nn_all = []

    for i in range(50):
        #data
        u_const=u_data[i].squeeze()
        u_in=u_const[0]
        u_fin=u_const[-1]
        u_in_full=u_data_full[i, 0]
        logs['u_const'].append(u_const)

        f_const=f_data[i].squeeze()
        logs['f_const'].append(f_const)

        f_nn, t_nn=burgers_nn_control(u_in.cuda(), u_fin.cuda(), int(s/2), args.reward_f, model_f=load_model_f, model_u=load_model_u, model_trans=load_model_trans, epochs=epochs, lr=lr, Nt=num_t, dt=delta_t, dx=delta_x, lamb1=args.lamb1, lamb2=args.lamb2, gtol=gtol, ftol=ftol)
        logs['f_nn'].append(f_nn)
        logs['t_nn'].append(t_nn)
        with HiddenPrints():
            u_nn_=burgers_numeric_solve(u_in_full.unsqueeze(0).cuda(), f_nn.unsqueeze(0), visc=0.01, T=1.0, dt=1e-4, num_t=num_t)
        u_nn_=u_nn_.squeeze()
        u_nn=torch.cat((u_nn_[:,:int(s/4)], u_nn_[:,int(3*s/4):]), -1)
        u_nn_surrogate=burgers_nn_solve(u_in.cuda(), f_nn.unsqueeze(0), load_model_f, load_model_u, load_model_trans, int(s/2), num_t)
        u_nn_surrogate_all.append(u_nn_surrogate.detach().cpu().numpy())
        u_nn_all.append(u_nn[1:].cpu().numpy())        
        logs['u_nn'].append(u_nn)
        logs['u_nn'].append(u_nn)
        er_u_nn, er_f_nn, er_uf_nn = objective_value(u_nn[-1].cpu().numpy(), f_nn.cpu().numpy(), u_fin.numpy(), s/2, args.reward_f)
        logs['loss_obj_nn'].append(er_u_nn)
        logs['loss_f_nn'].append(er_f_nn)
        logs['loss_all_nn'].append(er_uf_nn)

        print("No.{}, Neural : T={:3.2f}, J={:1.4f}+{:1.4f}={:1.4f}".format(i, t_nn, er_u_nn, er_f_nn, er_uf_nn)) 
    
    mean_loss_obj_nn = np.mean(np.array(logs['loss_obj_nn']))
    mean_loss_f_nn = np.mean(np.array(logs['loss_f_nn']))
    mean_loss_all_nn = np.mean(np.array(logs['loss_all_nn']))
    u_nn_surrogate_all = torch.tensor(np.array(u_nn_surrogate_all))
    u_nn_all = torch.tensor(np.array(u_nn_all))
    mean_state_loss = mse(u_nn_surrogate_all, u_nn_all)
    mean_relative_state_loss = torch.norm(u_nn_surrogate_all - u_nn_all) / torch.norm(u_nn_all)
    print("Neural    : J={:1.4f}+{:1.4f}={:1.4f}, state loss:{:1.8f}, relative state loss:{:1.5f}"\
            .format(mean_loss_obj_nn, mean_loss_f_nn, mean_loss_all_nn, mean_state_loss.item(), mean_relative_state_loss.item())) 
    
    #save
    torch.save({'logs': logs,
                'args': args
                }, 'logs/test_pob_fctr_{}'.format(date))
