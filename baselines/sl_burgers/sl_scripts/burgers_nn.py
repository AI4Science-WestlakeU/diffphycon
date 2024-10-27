import torch
import torch.nn.functional as F

import time

from .burgers_numeric import *
from .utils import *


RESCALER = RESCALER_1e5 = 6.4519
def burgers_nn_control(u0, uT, s, reward_f, model_f, model_u, model_trans, epochs, lr, Nt, dt, dx, lamb1, lamb2, gtol=1e-5, ftol=1e-6):
    u0=u0/RESCALER
    uT=uT/RESCALER
    epsilon=0.01*torch.rand((Nt,1,128))
    f_optim=torch.zeros((Nt,1,128))+epsilon
    f_optim=f_optim.cuda()/RESCALER
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
                if s < 128:
                    f_latent=f_latent[:, :128]+f_latent[:, 128:]
                nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(1,8,s//4), f_latent.view(1,8,s//4)), 1))
                loss3+=rel_error(f_rec, f_optim[i])
                u_t_latent=model_u.down(model_u.up(nn_u_t_dt_latent).reshape(-1,1,s)) 
            trans = model_u.up(nn_u_t_dt_latent)
            
            loss3=loss3/Nt*RESCALER**2
            loss1=torch.sum((trans.squeeze()-uT)**2)/s*RESCALER**2
            loss2=torch.sum(f_optim.squeeze()**2)*RESCALER**2

            loss=lamb1*(loss1+reward_f*loss2)+lamb2*loss3
            # print(loss1.item(), loss2.item(), loss3.item())
            # print(torch.sum((f_optim*RESCALER).squeeze()**2))
            loss.backward()
            return loss
        optimizer.step(closure)
    end = time.time()
    return f_optim.squeeze().detach()*RESCALER, end-start



RESCALER = RESCALER_1e5 = 6.4519
def burgers_nn_control_2(u0, uT, s, reward_f, model_f, model_u, model_trans, epochs, lr, Nt, dt, dx, lamb1, lamb2, gtol=1e-5, ftol=1e-6):
    u0=u0/RESCALER
    uT=uT/RESCALER
    epsilon=0.01
    f_optim=torch.zeros((Nt,1,128))+epsilon
    f_optim=f_optim.cuda()/RESCALER
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
                if s < 128:
                    f_latent=f_latent[:, :128]+f_latent[:, 128:]
                nn_u_t_dt_latent = model_trans(torch.cat((u_t_latent.view(1,8,s//4), f_latent.view(1,8,s//4)), 1))
                loss3+=rel_error(f_rec, f_optim[i])
                u_t_latent=model_u.down(model_u.up(nn_u_t_dt_latent).reshape(-1,1,s)) 
            trans = model_u.up(nn_u_t_dt_latent)
            
            loss3=loss3/Nt*RESCALER**2
            loss1=torch.sum((trans.squeeze()-uT)**2)/s*RESCALER**2
            loss2=torch.sum(f_optim.squeeze()**2)*RESCALER**2

            loss=lamb1*(loss1+reward_f*loss2)+lamb2*loss3
            # print(loss1.item(), loss2.item(), loss3.item())
            # print(torch.sum((f_optim*RESCALER).squeeze()**2))
            loss.backward()
            return loss
        optimizer.step(closure)
    end = time.time()
    return f_optim.squeeze().detach()*RESCALER, end-start
