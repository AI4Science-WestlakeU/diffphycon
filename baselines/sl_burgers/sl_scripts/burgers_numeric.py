import numpy as np
import torch
import torch.nn.functional as F

import math
from collections import OrderedDict
import time

from .diff_matrices import Diff_mat_1D

def burgers_numeric_solve(u0, f, visc, T, dt=1e-4, num_t=10, mode=None):
    if mode!='const':
        assert f.size()[1]==num_t, 'check number of time interval'
    else:
        f=f.unsqueeze(1).repeat(1,num_t,1)
    
    # u0: (Nu0,s)
    # f: (Nf,Nt,s)

    #Grid size
    s = u0.size()[-1]
    
    Nu0 = u0.size()[0]
    Nf = f.size()[0]
    Nt = f.size()[1]
    
    xmin = 0.0; xmax = 1.0
    delta_x = (xmax-xmin)/(s+1)

    #Number of steps to final time
    steps = math.ceil(T/dt)

    u = u0.reshape(Nu0,1,s)
    u = F.pad(u, (1,1))
    f = f.reshape(1,Nf,Nt,s)
    f = F.pad(f, (1,1))
    
    #Record solution every this number of steps
    record_time = math.floor(steps/Nt)
    
    D_1d, D2_1d = Diff_mat_1D(s+2)   
    #remedy?
    D_1d.rows[0] = D_1d.rows[0][:2]
    D_1d.rows[-1] = D_1d.rows[-1][-2:]
    D_1d.data[0] = D_1d.data[0][:2]
    D_1d.data[-1] = D_1d.data[-1][-2:]
    
    D2_1d.rows[0] = D2_1d.rows[0][:3]
    D2_1d.rows[-1] = D2_1d.rows[-1][-3:]
    D2_1d.data[0] = D2_1d.data[0][:3]
    D2_1d.data[-1] = D2_1d.data[-1][-3:]
    
    t_sys_ind = list(D_1d.rows)
    t_sys = torch.FloatTensor(np.stack(D_1d.data)/(2*delta_x)).to(u0.device)
    d_sys_ind = list(D2_1d.rows)
    d_sys = torch.FloatTensor(visc*np.stack(D2_1d.data)/delta_x**2).to(u0.device)
    
    #Saving solution and time
    sol = torch.zeros(Nu0, Nf, s, Nt, device=u0.device)
    
    #Record counter
    c = 0
    #Physical time
    t = 0.0
    f_idx=-1
    for j in range(steps):
        u = u[...,1:-1]
        u = F.pad(u, (1,1))
        
        u_s = u**2
        transport = torch.einsum('bcsi,si->bcs', u_s[...,t_sys_ind], t_sys)
        diffusion = torch.einsum('bcsi,si->bcs', u[...,d_sys_ind], d_sys)
        if j % record_time == 0:
            f_idx+=1
        u = u + dt * (-(1/2)*transport + diffusion + f[:,:,f_idx,:])
        
        #Update real time (used only for recording)
        t += dt

        if (j+1) % record_time == 0:

            #Record solution and time
            sol[...,c] = u[...,1:-1]
            c += 1

    sol=sol.permute(0,1,3,2) #(Nu0,Nf,Nt,s)
    trajectory = torch.cat((u0.reshape(Nu0,1,1,s).repeat(1,Nf,1,1),sol),dim=2)
    return trajectory

