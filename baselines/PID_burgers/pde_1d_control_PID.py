#!/usr/bin/env python
# coding: utf-8

# In[39]:


try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass


# In[40]:


import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import h5py
import random
from copy import copy
from datetime import datetime
import argparse
from IPython import embed
from generate_burgers import burgers_numeric_solve
import pdb
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import tqdm
import termcolor

parser = argparse.ArgumentParser(description='Generating PDE data')
parser.add_argument('--experiment', type=str, default='burgers_pid',
                    help='Experiment for which data should create for')
parser.add_argument('--exp_path', default='/user/project/pde_gen_control', type=str, help='experiment folder')
parser.add_argument('--date_time', default='2023-12-01_test', type=str, help='experiment date')
parser.add_argument('--model_type', default='PID', type=str, help='model type.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='Used device id')
parser.add_argument('--num_f', default=500, type=int,
                    help='the number of force data')
parser.add_argument('--num_u0', default=60, type=int, 
                    help='the number of initial data')
parser.add_argument('--train_samples', type=int, default=24000,
                    help='Samples in the training dataset')
# parser.add_argument('--valid_samples', type=int, default=0,
#                     help='Samples in the validation dataset')
parser.add_argument('--test_samples', type=int, default=6000,
                    help='Samples in the test dataset')
parser.add_argument('--log', type=eval, default=False,
                    help='pip the output to log file')
parser.add_argument('--max_iter_steps', type=int, default=10,
                    help='max iter steps for tuning params of PID')
parser.add_argument('--max_training_iters', type=int, default=10,
                    help='max number of iters for tuning params of PID')
parser.add_argument('--save_iters', type=int, default=5,
                    help='save weight each save_iters iters')
parser.add_argument('--model_mode', type=str, default='train',
                    help='train or eval')
parser.add_argument('--model_weight_path', type=str, default='train',
                    help='path of model weight')
parser.add_argument('--dataset_path', type=str, default='/user/project/pde_gen_control/dataset/dataset_control_burgers/free_u_f_1e5',
                    help='path of dataset path')
parser.add_argument('--train_batch_size', type=int, default=16,
                    help='batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='batch size for training')
parser.add_argument('--coef_f_loss', type=float, default=0,
                    help='coef_f_loss')
parser.add_argument('--is_partially_controllable', type=int, default=0,
                    help='0: fully controllable; 1. partially_controllable')
parser.add_argument('--simulation_method', type=str, default="solver",
                    help='solver or surrogate_model')
parser.add_argument('--pde_1d_surrogate_model_checkpoint', type=str, default="/user/project/pde_gen_control/results/2023-12-03_1d_surrogate_model",
                    help='1d pde surrogate_model checkpoint path')
parser.add_argument('--f_max', type=float, default=1,
                    help='1d pde surrogate_model checkpoint path')
parser.add_argument('--is_partially_observable', type=int, default=0,
                    help='0: fully observable; 1. partially_observable')
# In[41]:

#MIMO PID Controller without decouplerd
class PID_Controller_MIMO:
    def __init__(self,Kp=None,Ki=None,Kd=None,ns=128):
        '''
        init input :
            Kp tensor [ns]
            Ki tensor [ns] 
            Kd tensor [ns] 
        '''
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
        self.err_sum=0
        self.last_error=0
        self.ns=ns
    def init_parameters(self):
        self.Kp=torch.randn((self.ns,))
        self.Ki=torch.randn((self.ns,))
        self.Kd=torch.randn((self.ns,))


    def PID_controller(self,error):
        '''
        input:
            error: tensor [batch_size,ns]
        return:
            f: tensor [batch_size]
        '''   
        delta_error=error-self.last_error
        self.err_sum=self.err_sum+error
        f=error*self.Kp+self.err_sum*self.Ki+delta_error*self.Kd

        return f
    def save_PID_parameters(self,path):
        #save PID parameters
        torch.save(self.Kp, path+'/Kp.pt')
        torch.save(self.Kp, path+'/Ki.pt')
        torch.save(self.Kp, path+'/Kd.pt')
    def load_PID_parameters(self,path):
        self.Kp=torch.load(path+"/Kp.pt")
        self.Ki=torch.load(path+"/Ki.pt")
        self.Kd=torch.load(path+"/Kd.pt")


# In[42]:

class Printer(object):
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string
p = Printer(n_digits=6)

def pid_controller(PID=None,error=None,last_error=None,error_sum=None):
    control=PID[:,0,:]*error+PID[:,1,:]*error_sum+PID[:,2,:]*(error-last_error)
    return control


class Network(nn.Module):
    def __init__(self, ns=128):
        super(Network, self).__init__()

        # Define the linear layers and activation functions
        self.ns=ns
        self.fc1 = nn.Linear(ns, 2 * ns)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2 * ns, 3 * ns)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(2 * ns, 3 * ns)

    def forward(self, x):
        # Forward pass through the layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x=self.fc3(x)

        # Reshape the output to the desired shape [batch_size, 3, ns]
        x = x.view(-1, 3, self.ns)

        return x

class cNet(nn.Module):
    def __init__(self, ns=128):
        super(cNet, self).__init__()

        # Define 1D convolutional layers and activation functions
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.Softsign()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.Softsign()

        # Define the linear layers and activation functions
        self.fc1 = nn.Linear(32 * ns, 8 * ns)
        self.act3 = nn.Softsign()
        self.fc2 = nn.Linear(8 * ns, 3 * ns)
        self.act4 = nn.Softsign()

        self.ns = ns

    def forward(self, x):
        # Add a dummy dimension for the channel in 1D convolution
        x = x.unsqueeze(1)

        # Forward pass through the 1D convolutional layers
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        # Reshape the output for the linear layers
        x = x.view(-1, 32 * self.ns)

        # Forward pass through the linear layers
        x = self.fc1(x)
        x = self.act3(x)
        # # x=x*100
        x = self.fc2(x)
        x = self.act4(x)
        x=x*10

        # Reshape the output to the desired shape [batch_size, 3, ns]
        x = x.view(-1, 3, self.ns)

        return x

class Controller(nn.Module):
    def __init__(self,model,loss_type= 'l1',control_mask=None,obsereved_mask=None):
        super(Controller, self).__init__()
        self.loss_type=loss_type
        self.model=model
        self.last_error=0
        self.error_sum=0
        self.control_mask=control_mask
        self.obsereved_mask=obsereved_mask
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    def forward(self,error):
        if self.obsereved_mask!=None:            
            error=error*self.obsereved_mask
            error_ob=torch.cat([error[:,:32],error[:,-32:]],dim=-1)
            PID=torch.zeros((error.shape[0],3,error.shape[-1]),device=error.device)
            temp=self.model(error_ob)
            PID[:,:,:32]=temp[:,:,:32]
            PID[:,:,-32:]=temp[:,:,-32:]
        else:
            PID=self.model(error)
        self.error_sum=self.error_sum+error
        control=pid_controller(PID=PID,error=error,last_error=self.last_error,error_sum=self.error_sum)
        if self.control_mask!=None:
            control=control*self.control_mask
        self.last_error=error
        return control
import matplotlib.backends.backend_pdf

def metric(u_controlled: torch.Tensor,u_target: torch.Tensor, f: torch.Tensor, target='final_u'):
    '''
    Evaluates the control based on the state deviation and the control cost.
    Note that f and u should NOT be rescaled. (Should be directly input to the solver)

    Arguments:
        u_target:
            Ground truth states
            size: (batch_size, Nt, Nx) (currently Nt = 11, Nx = 128)
        f: 
            Generated control force
            size: (batch_size, Nt - 1, Nx) (currently Nt = 11, Nx = 128)
    
    Returns:
        J_actual:
            Deviation of controlled u from target u for each sample, in MSE.
            When target is 'final_u', evaluate only at the final time stamp
            size: (batch_size)
        
        control_energy:
            Cost of the control force for each sample, in sum of square.
            size: (bacth_size)
    '''
    
    assert len(u_target.size()) == len(f.size()) == 3

    # u_controlled = burgers_numeric_solve_free(u_target[:, 0, :], f, visc=0.01, T=1.0, dt=1e-4, num_t=10)

    # eval J_actual
    if target == 'final_u':
        J_actual = (u_controlled[:, -1, :] - u_target[:, -1, :]).square().mean(-1)
    else:
        raise ValueError('Undefined target to evaluate')
    
    control_energy = f.square().sum((-1, -2))

    return J_actual, control_energy
def plot_result(u0,ut,ud,ut_free,f,ut_gd,ut_free_gd,path):
    '''
    Args:
        u_0: target u at t=0 [batch_size,ns]
        u_t: u at t=T (final state) [batch_size,n_steps=10,ns]
        u_d: target u at t=T (final state) [batch_size,ns]
        u_t_gd: u at t=T (final state) [batch_size,n_steps=10,ns]
        f  :force to control u [[batch_size,n_steps=10,ns]
        free means no force control
    '''
    x_pos = np.linspace(0, 1, 128)
    pdf = matplotlib.backends.backend_pdf.PdfPages(path+f"/inference_results.pdf")
    fontsize = 16
    J_actual, control_energy=metric(u_controlled=ut_gd,u_target=ud.reshape(ud.shape[0],1,-1), f=f, target='final_u')
    J_actual_mean=J_actual.mean()
    print(control_energy)
    control_energy_mean=control_energy.mean(0)
    if ut!=None:
        RMSE=(ut_gd-ut).square().mean(-1)
        RMSE=RMSE.mean(-1)
        ut=ut[:,-1]
    ut_gd=ut_gd[:,-1]
    ut_free_gd=ut_free_gd[:,-1]
    for i in range(16):
        i=i*1
        fig = plt.figure(figsize=(18,15))
        if ut!=None:
            plt.plot(x_pos, ut[i], color="red", linestyle="-",label=r"u_t prediction")
        plt.plot(x_pos, ut_gd[i], color="yellow", linestyle="-",label=r"u_t ground truth")
        if ut!=None:
            plt.title(f"J_actual_mean: {J_actual_mean:.5f} J_actual: {J_actual[i]:.5f} \n control_energy_mean: {control_energy_mean:.5f} control_energy: {control_energy[i]:.5f} \n MSE:{RMSE.mean():.5f} SE:{RMSE[i]:.5f}", fontsize=fontsize)
        else:
            plt.title(f"J_actual_mean: {J_actual_mean:.5f} J_actual: {J_actual[i]:.5f} \n control_energy_mean: {control_energy_mean:.5f} control_energy: {control_energy[i]:.5f} \n ", fontsize=fontsize)
        plt.legend()
        plt.tick_params(labelsize=fontsize)
        pdf.savefig(fig)
        # plt.show()
    pdf.close()
from matplotlib import colors
cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}
cmap = colors.LinearSegmentedColormap('custom', cdict)
def plot_result_process(u_gt_list, u0_list, uf_list,path):
    '''
    Args:
        x: concated u and f (the output state of DDPM model)  
        x_gt: concated u_gt and f_gt
        u_0: target u at t=0
        u_f: target u at t=T (final state)
    '''

    
    pdf = matplotlib.backends.backend_pdf.PdfPages(path+f"/inference_results_process.pdf")
    for j in range(u0_list.shape[0]):
        fig, ax = plt.subplots(1, 1, figsize=(20,12))
        u0=u0_list[j]
        uf=uf_list[j]
        u_gt=u_gt_list[j]
        fig.set_size_inches(6, 4)
        x_pos = np.linspace(0, 1, 128)
        for i in range(10):
            if i in [0, 3, 7, 9]:
                # ax.plot(x_pos, u[i + 1, :], label=f"Diffusion $u_{{t={i + 1}}}$", color=cmap(i / 10))
                ax.plot(x_pos, u_gt[i + 1, :], label=f'Simulation, $u_{{t={i + 1}}}$', color=cmap((i) / 10), ls='--')
            else:
                # ax.plot(x_pos, u[i + 1, :], color=cmap(i / 10))
                ax.plot(x_pos, u_gt[i + 1, :], color=cmap((i) / 10), ls='--')
        ax.plot(x_pos, u0, label='Target $u_{t=0}$', color='blue', ls='-.')
        ax.plot(x_pos, uf, label='Target $u_{t=10}$', color='green', ls='-.')
        ax.legend(ncol=2, loc='center', bbox_to_anchor=(1.6, 0.5))
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        pdf.savefig(fig,ax=ax)
    pdf.close()

# In[43]:


from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader
def cycle(dl):
    while True:
        for data in dl:
            yield data
class train(object):
    def __init__(
            self,
            controller=None,
            max_training_iters=10,
            max_eval_iters=10,
            Ud=None,
            U0=None,
            max_iter_steps=100,
            mode='random',
            lr=1e-4,
            save_iters=2,
            exp_path=None,
            learn_steps=100,
            dataset=None,
            train_batch_size=16,
            device=None,
            coef_f_loss=0,
            simulation_method="solver",
            simu_surrogate_model=None
    ):
        self.controller=controller
        self.Ud=Ud
        self.max_training_iters=max_training_iters
        self.max_eval_iters=max_eval_iters
        self.U0=U0
        self.max_iter_steps=max_iter_steps
        self.mode=mode
        self.save_iters=save_iters
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=lr)
        self.exp_path=exp_path
        self.learn_steps=learn_steps
        self.device=device
        self.train_batch_size=train_batch_size
        dl=DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers =24)
        self.dl=cycle(dl)
        self.coef_f_loss=coef_f_loss
        self.simulation_method=simulation_method
        self.simu_surrogate_model=simu_surrogate_model
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=40000, gamma=0.5)
    def train(self):
        self.controller.model.train()
        loss_list=torch.zeros((self.max_training_iters)).flatten()
        trajectory=None
        data = next(self.dl).to(self.device)
        self.U0=data[:,0]
        self.Ud=data[:,10]
        print(self.max_training_iters)
        for j in range(self.max_training_iters):
            if j==0:
                p.print(f"test_start  {j}", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
            self.controller.error_sum=0
            self.controller.last_error=0
            loss_sum=0
            data = next(self.dl).to(self.device)
            for i in tqdm.trange(self.max_iter_steps):
                if i==0:
                    if j%10==0:
                        data = next(self.dl).to(self.device)
                    self.U0=data[:,0] #data [batch_size,num_timesteps,ns]
                    self.Ud=data[:,10]
                    ut=self.U0
                    control=self.controller(ut-self.Ud)
                    control=control.reshape(control.shape[0],1,-1)
                else:
                    # u0=ut.clone().detach()
                    control=self.controller(ut-self.Ud)
                    control=control.reshape(control.shape[0],1,-1)##[batch_size,1,ns]
                control=torch.clamp(control, -args.f_max, args.f_max)
                #burgers_numeric_solve 
                #u0 [batch_size,ns]
                # pdb.set_trace()
                if self.simulation_method=="solver":
                    trajectory=burgers_numeric_solve(ut, control, visc=0.01, T=1e-1, dt=1e-4, num_t=1, mode=self.mode)
                    ut=trajectory[torch.arange(self.train_batch_size),torch.arange(self.train_batch_size),1]##[batch_size,ns
                elif self.simulation_method=="surrogate_model":
                    # pdb.set_trace()
                    if args.is_partially_observable==1:
                        ut_partial=torch.cat([ut[:,:32],ut[:,-32:]],dim=-1)
                        ut_partial=self.simu_surrogate_model.simulation(ut=ut_partial,ft=control)
                        ut_partial=ut_partial.reshape(ut_partial.shape[0],ut_partial.shape[-1])
                        ut[:,:32]=ut_partial[:,:32]
                        ut[:,-32:]=ut_partial[:,-32:]
                    else:
                        ut=self.simu_surrogate_model.simulation(ut=ut,ft=control)
                        ut=ut.reshape(ut.shape[0],ut.shape[-1])
                if controller.obsereved_mask!=None:
                    # pdb.set_trace()
                    # test=self.controller.loss_fn(self.Ud[:,:32],ut[:,:32])+self.controller.loss_fn(self.Ud[:,-32:],ut[:,-32:])
                    loss=self.controller.loss_fn(self.Ud*controller.obsereved_mask,ut*controller.obsereved_mask)+control.mean().abs()*self.coef_f_loss
                else:
                    loss=self.controller.loss_fn(self.Ud,ut)+control.mean().abs()*self.coef_f_loss
                # pdb.set_trace()
                loss=loss.mean()
                if not torch.isnan(ut).any().item():
                    loss_sum=loss_sum+loss
            loss_sum.backward(retain_graph=True)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list[j]=loss_sum.clone().detach()
            if j%(self.save_iters-1)==0:
                print(f"training {j} iters,loss_sum: {loss_sum}")
                torch.save(self.controller.model.state_dict(), self.exp_path+f'/model_weights-{j}.pth')
            if j==0:
                p.print(f"test_end  {j}", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
        numpy_data = loss_list.to("cpu").detach().numpy()
        # draw loss_list
        plt.figure()
        x=np.linspace(0,len(numpy_data),len(numpy_data))
        plt.plot(x, numpy_data)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('loss_list')
        plt.grid(True)
        plt.savefig(self.exp_path+"/loss_list.png")
        # plt.show()
        np.save(self.exp_path+'/loss_lis.npy', numpy_data)
    @torch.no_grad()
    def eval(self,weigh_path):
        self.controller.model.eval()
        #load weight
        self.controller.model.load_state_dict(torch.load(weigh_path))
        loss_list=torch.zeros((self.max_eval_iters*self.max_iter_steps)).flatten()
        trajectory=None
        for j in tqdm.trange(self.max_eval_iters):
            print(f"eval {j} iters")
            self.controller.error_sum=0
            self.controller.last_error=0
            loss_sum=0
            for i in tqdm.trange(self.max_iter_steps):
                if i==0:
                    if self.U0!=None:
                        u0=self.U0
                    else:
                        u0=torch.randn_like(self.Ud)
                    control=self.controller.forward(u0-self.Ud)
                    control=self.control.reshape(control.shape[0],1,-1)
                else:
                    u0=ut.clone().detach()
                    control=self.controller.forward(u0-self.Ud)
                    control=control.reshape(control.shape[0],1,-1)##[batch_size,1,ns]
                #burgers_numeric_solve 
                #u0 [batch_size,ns]
                # pdb.set_trace()
                trajectory=burgers_numeric_solve(u0, control, visc=0.01, T=1e-4, dt=1e-4, num_t=1, mode=self.mode)
                ut=trajectory[:,0,1,:]##[batch_size,ns
                loss=self.controller.loss_fn(self.Ud,ut)
                loss_list[i+j*self.max_iter_steps]=loss.clone().detach()
        numpy_data = loss_list.to("cpu").detach().numpy()
        # draw loss_list
        plt.figure()
        x=np.linspace(0,len(numpy_data),len(numpy_data))
        plt.plot(x, numpy_data)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('loss_list')
        plt.grid(True)
        plt.savefig(self.exp_path+"/eval_loss_list.png")
        # plt.show()
        np.save(self.exp_path+'/eval_loss_lis.npy', numpy_data)


# In[45]:
if __name__=="__main__":
    args = parser.parse_args()
    is_jupyter = False
    device=f"cuda:{args.gpuid}"
    from data_burgers_1d import Burgers1DSimple, Burgers
    train_dataset = Burgers1DSimple(
        dataset="burgers",
        input_steps=1,
        output_steps=10,
        time_interval=1,
        is_y_diff=False,
        split="train",
        transform=None,
        pre_transform=None,
        verbose=False,
        root_path =args.dataset_path ,
        device='cuda',
        rescaler=1
    )
    print(args)
    if args.model_mode=="train":
        if args.is_partially_controllable==1:
            control_mask=torch.ones((args.train_batch_size,128),device=device)
            # control_mask[:,:32]=control_mask[:,:32]*0
            # control_mask[:,-32:]=control_mask[:,-32:]*0
            control_mask[:,32:96]=control_mask[:,32:96]*0

        else:
            control_mask=None
        if args.is_partially_observable==1:
            obsereved_mask=torch.ones((args.train_batch_size,128),device=device)
            # control_mask[:,:32]=control_mask[:,:32]*0
            # control_mask[:,-32:]=control_mask[:,-32:]*0
            obsereved_mask[:,32:96]=obsereved_mask[:,32:96]*0
        else:
            obsereved_mask=None
        u0=torch.rand(args.num_u0,128).to(device)
        ud=torch.rand(args.num_u0,128).to(device)
        exp_path=args.exp_path+"/results/"+args.date_time+"/"+args.experiment
        if args.simulation_method=="solver":
            simu_surrogate_model=None
        elif args.simulation_method=="surrogate_model":
            from model.pde_1d_surrogate_model.burgers_operator import Simu_surrogate_model
            milestone=500
            if args.is_partially_observable==1:
                s_ob=64
            else:
                s_ob=128
            simu_surrogate_model=Simu_surrogate_model(path=args.pde_1d_surrogate_model_checkpoint,device=device,s_ob=s_ob,milestone=milestone)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            print(f"{exp_path} created")
        else:
            print(f"{exp_path} exists")
        if args.is_partially_observable==1:
            model=cNet(ns=64)
        else:
            model=cNet()
        model=model.to(device)
        controller=Controller(model=model,control_mask=control_mask,obsereved_mask=obsereved_mask)
        trainer=train(
            controller=controller,
            U0=u0,
            Ud=ud,
            max_iter_steps=args.max_iter_steps,
            max_training_iters=args.max_training_iters,
            save_iters=args.save_iters,
            exp_path=exp_path,
            learn_steps=5,
            device=device,
            dataset=train_dataset,
            train_batch_size=args.train_batch_size,
            lr=args.lr,
            coef_f_loss=args.coef_f_loss,
            simulation_method=args.simulation_method,
            simu_surrogate_model=simu_surrogate_model,
        )
        trainer.train()