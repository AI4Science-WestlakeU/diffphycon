import torch
import torch.nn.functional as F

import time

from .burgers_numeric import *
from .utils import *

def burgers_nn_control(u0, uT, model_f, model_u, model_trans, epochs, lr, Nt, dt, dx, lamb1, lamb2, gtol=1e-5, ftol=1e-6):
    s = u0.size()[-1]
    epsilon=0.01
    f_optim=torch.zeros((Nt,1,s))+epsilon
    f_optim.requires_grad = True
    optimizer = torch.optim.LBFGS([f_optim], line_search_fn='strong_wolfe', lr=lr, tolerance_grad=gtol, tolerance_change=ftol)
    
    start = time.time()
    for epoch in range(1, epochs+1):
        def closure():
            optimizer.zero_grad()
            loss3=0
            u_t_data=u0.reshape(1,1,s)
            u_t_rec, u_t_latent = model_u(u_t_data)
            for i in range(Nt):
                f_rec, f_latent = model_f(f_optim[[i]])
                nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(1,8,s//4), f_latent.view(1,8,s//4)), 1))
                loss3+=rel_error(f_rec, f_optim[i])
                u_t_latent=model_u.down(model_u.up(nn_u_t_dt_latent).reshape(-1,1,s)) 
            trans = model_u.up(nn_u_t_dt_latent)
            
            alpha=0.01    
            loss3=loss3/Nt
            loss1=(1/2)*torch.sum((trans.squeeze()-uT)**2)*dx
            loss2=(alpha/2)*torch.sum(f_optim.squeeze()**2)*dx*dt

            loss=lamb1*(loss1+loss2)+lamb2*loss3
            loss.backward()
            return loss
        optimizer.step(closure)
    end = time.time()
    return f_optim.squeeze().detach(), end-start