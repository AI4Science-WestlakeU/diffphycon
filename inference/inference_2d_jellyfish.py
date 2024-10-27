import argparse
from collections import OrderedDict
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import datetime
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import pdb
import pickle
import pprint as pp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dataset.data_2d import Jellyfish
from torch.autograd import grad
import tqdm
from accelerate import Accelerator
from pathlib import Path
from diffusion.diffusion_2d_jellyfish import Unet, ForceUnet, GaussianDiffusion, Trainer
from model.video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D
from matplotlib.backends.backend_pdf import PdfPages
from filepath import JELLYFISH_DATA_PATH, JELLYFISH_RESULTS_PATH
from IPython import embed

normalization_filename = os.path.join(JELLYFISH_DATA_PATH, "train_data/normalization_max_min.pkl") 
normdict = pickle.load(open(normalization_filename, "rb"))
p_max = normdict["p_max"]
p_min = normdict["p_min"]
vx_max = normdict['vx_max']
vx_min = normdict['vx_min']
vy_max = normdict['vy_max']
vy_min = normdict['vy_min']

# unnormalize_state range from [-1, 1] to [p_min, p_max]
def unnormalize_state(pressure):
    return (0.5 * pressure + 0.5) * (p_max - p_min) + p_min

# normalize_state range
def normalize_batch(state_full):
    state_full[:,0] = ((state_full[:,0,:,:] - vx_min) / (vx_max - vx_min) - 0.5) * 2
    state_full[:,1] = ((state_full[:,1,:,:] - vy_min) / (vy_max - vy_min) - 0.5) * 2
    state_full[:,2] = ((state_full[:,2,:,:] - p_min) / (p_max - p_min) - 0.5) * 2
    return state_full  

def reg_theta(theta):
    """
    input: theta of shape [batch_size, T]
    output: work of shape [batch_size]
    
    compute R(theta) = \sum_{t=0}^{T-2}(theta_{t+1} - theta_t) ** 2
    """
    theta_t = theta[:, :-1] # theta_{t}: [batch_size, T-1]
    theta_t_1 = theta[:, 1:] # theta_{t+1}: [batch_size, T-1]

    reg = torch.sum((theta_t_1 - theta_t) * (theta_t_1 - theta_t), dim=1) # [batch_size]
    
    return reg

def clip_loss(tensor, min_value, max_value):
    clipped_tensor = torch.clamp(tensor, min_value, max_value)
    
    loss = (tensor - clipped_tensor).abs().mean()
    
    return loss

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
def get_CosineAnnealingLR(num_iters,lr_max=0.0001,eta_min=0.0000001):
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=lr_max)
    total_epochs = num_iters
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    learning_rates = []
    coef_grad_list=torch.zeros(num_iters)
    for epoch in range(total_epochs):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        coef_grad_list[epoch]=current_lr
    return coef_grad_list

def force_fn(x, bd_0, force_model, bd_updater, args):
    if args.only_vis_pressure:
        state, theta_expand = x[:,:,:1,:,:], x[:,:,-1,:,:]
    else:
        state, theta_expand = x[:,:,:3,:,:], x[:,:,3,:,:]
    state.requires_grad_()
    theta_expand.requires_grad_()
    theta = torch.mean(torch.mean(theta_expand, dim=3), dim=2)
    if args.only_vis_pressure:
        pressure = state[:, :, 0, :, :] # state: [batch, 20, 6, 128, 128], pressure: [batch, 20, 128, 128]
    else:
        pressure = state[:, :, 2, :, :] # state: [batch, 20, 6, 128, 128], pressure: [batch, 20, 128, 128]
    pressure = unnormalize_state(pressure)
    pred_bd = bd_updater(
        bd_0.reshape(bd_0.shape[0] * bd_0.shape[1], bd_0.shape[2], bd_0.shape[3], bd_0.shape[4]), 
        theta.reshape(theta.shape[0] * theta.shape[1])
    )
    pred_bd = pred_bd.reshape(bd_0.shape) # output pred_bd: [batch, 20, 3, 128, 128]
    input = torch.cat((pressure.unsqueeze(2), pred_bd), dim=2) # input: [batch, 20, 4, 128, 128]
    input = input.reshape(input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]) # output input: [batch * 20, 4, 128, 128]
    force = force_model(input) # output force: [batch_size * 20, 2]
    force = force.reshape(state.shape[0], state.shape[1]) # output force: [batch_size, 20]
    weight = torch.FloatTensor(range(force.shape[1], 0, -1)).to(args.device).expand(force.shape[0], force.shape[1]) # weight: [batch_size, 21]
    average_velocity = torch.mean(force * weight, dim=1) # average velocity over time steps
    reg = reg_theta(theta)
    guidance = -average_velocity + args.reg_ratio * reg # [batch_size]
    
    grad_state, grad_theta = grad(guidance, [state, theta_expand], grad_outputs=torch.ones_like(guidance))

    return grad_state, grad_theta


# will be push into utils
def mask_denoise(tensor, thre=0.5):
    binary_tensor = torch.where(tensor > thre, torch.tensor(1), torch.tensor(0))
    return binary_tensor


def load_model(args):
    # load main model of each inference method
    if args.inference_method == 'DDPM':
        inp_dim = 5 if args.only_vis_pressure else 7 # use state of velocity and pressure (3), and boundary mask and offset (3), and theta (1)
        out_dim = 2 if args.only_vis_pressure else 4 #joint of state and theta model
        #out_dim = 1 if args.only_vis_pressure else 3, # velocity, pressure # state model 
        model_joint = Unet3D_with_Conv3D(
            dim = 64,
            out_dim = out_dim,
            dim_mults = (1, 2, 4),
            channels= inp_dim 
        )
        print("number of parameters model_joint: ", sum(p.numel() for p in model_joint.parameters() if p.requires_grad))
        model_joint.to(args.device)
        # diffusion_states 
        diffusion_joint = GaussianDiffusion(
            model_joint,
            image_size = args.image_size,
            frames = args.frames, 
            cond_steps = args.cond_steps,
            timesteps = 1000,           # number of steps
            sampling_timesteps = args.sampling_timesteps,   # do not use ddim
            loss_type = 'l2',            # L1 or L2
            objective = 'pred_noise',
            backward_steps = args.backward_steps,
            backward_lr = args.backward_lr, # used in universal-backward sampling
            standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
            forward_fixed_ratio = args.forward_fixed_ratio, # used in universal forward sampling
            coeff_ratio_J = args.coeff_ratio_J, # used in gradien of J in standard-alpha sampling
            coeff_ratio_w = args.coeff_ratio_w, # used in predicted noise of p(w) standard-alpha sampling
            only_vis_pressure=args.only_vis_pressure,
            eval_2ddpm = False,
            device = args.device
        )
        # diffusion_states.eval()
        # load trainer
        trainer = Trainer(
            # diffusion_states,
            diffusion_joint,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            frames=args.frames,
            traj_len=40,
            ts=1,
            log_path=args.log_path,
            train_batch_size = args.batch_size,
            train_lr = 1e-4,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 1,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            results_path = args.diffusion_joint_model_path, # diffuse on both cond states, pred states and boundary
            amp = False,                       # turn on mixed precision
            calculate_fid = False             # whether to calculate fid during training
        )
        trainer.load(args.diffusion_joint_checkpoint) # 
        
        model_thetas = Unet3D_with_Conv3D(
            dim = 64,
            out_dim = 1, # theta
            dim_mults = (1, 2, 4),
            channels=5 if args.only_vis_pressure else 7 # use state of velocity and pressure as condition (3), and boundary mask and offset (3), and theta (1)
        )
        print("number of parameters model_thetas: ", sum(p.numel() for p in model_thetas.parameters() if p.requires_grad))
        model_thetas.to(args.device)
        diffusion_thetas = GaussianDiffusion(
            model_thetas,
            image_size = args.image_size,
            frames = args.frames, 
            cond_steps = args.cond_steps,
            timesteps = 1000,           # number of steps
            sampling_timesteps = args.sampling_timesteps,
            loss_type = 'l2',            # L1 or L2
            objective = 'pred_noise',
            backward_steps = args.backward_steps,
            backward_lr = args.backward_lr, # used in universal-backward sampling
            standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
            forward_fixed_ratio = args.forward_fixed_ratio, # used in universal forward sampling
            coeff_ratio_J = args.coeff_ratio_J, # used in gradien of J in standard-alpha sampling
            coeff_ratio_w = args.coeff_ratio_w, # used in predicted noise of p(w) standard-alpha sampling
            only_vis_pressure=args.only_vis_pressure,
            eval_2ddpm = False,
            device = args.device
        )
        diffusion_thetas.eval()
        trainer = Trainer(
            diffusion_thetas,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            frames=args.frames,
            traj_len=40,
            ts=1,
            log_path=args.log_path,
            train_batch_size = args.batch_size,
            train_lr = 1e-4,
            train_num_steps = 700000,         # total training steps
            gradient_accumulate_every = 1,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            results_path = args.diffusion_w_model_path, # diffuse on both cond states, pred states and boundary
            amp = False,                       # turn on mixed precision
            calculate_fid = False             # whether to calculate fid during training
        )
        trainer.load(args.diffusion_w_checkpoint) # 
        
        diffusion = GaussianDiffusion(
            [diffusion_joint.model, diffusion_thetas.model],
            image_size = args.image_size,
            frames = args.frames, 
            cond_steps = args.cond_steps,
            timesteps = 1000,           # number of steps
            sampling_timesteps = args.sampling_timesteps,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            loss_type = 'l2',            # L1 or L2
            objective = 'pred_noise',
            backward_steps = args.backward_steps,
            backward_lr = args.backward_lr, # used in universal-backward sampling
            standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
            forward_fixed_ratio = args.forward_fixed_ratio, # used in universal forward sampling
            coeff_ratio_J = args.coeff_ratio_J, # used in gradien of J in standard-alpha sampling
            coeff_ratio_w = args.coeff_ratio_w, # used in predicted noise of p(w) standard-alpha sampling
            only_vis_pressure=args.only_vis_pressure,
            eval_2ddpm = True,
            w_prob_exp = args.w_prob_exp,
            use_guidance_in_model_predictions = args.use_guidance_in_model_predictions,
            device = args.device
        )
        
    elif args.inference_method == 'SAC':
        cp = torch.load(args.sac_model_path, map_location=lambda storage, loc: storage)
        if args.only_vis_pressure:
            model = SAC_pob(args.image_size, 1, cp['args'])
        else:
            model = SAC(args.image_size, 1, cp['args'])
        model.policy.load_state_dict(cp['policy_state_dict']) 
    
    # load force_model
    force_model = ForceUnet(
        dim = args.image_size,
        out_dim=1,
        dim_mults = (1, 2, 4, 8),
        channels=4
    )
    force_model.load_state_dict(torch.load(args.force_model_checkpoint, map_location=args.device))
    force_model.to(args.device)

    # load boundary_updater_model
    bd_updater = Unet(
        dim = args.image_size,
        out_dim = 3,
        dim_mults = (1, 2, 4, 8),
        channels=3
    )
    bd_updater.load_state_dict(torch.load(args.boundary_updater_model_checkpoint, map_location=args.device))
    bd_updater.to(args.device)
    # define design function
    def design_fn(x, bd_0):
        grad_state, grad_theta = force_fn(x, bd_0, force_model, bd_updater, args)
    
        return torch.cat([grad_state, grad_theta.unsqueeze(2)], dim=2)
    
    if args.inference_method == 'DDPM':
        return force_model, diffusion, bd_updater, design_fn
    elif args.inference_method == 'SAC':
        return force_model, model, bd_updater, design_fn
    elif args.inference_method == 'MPC':
        return force_model, None, bd_updater, design_fn

from sim_ppl_2d import SurrogatePipeline, build_ppl, build_ppl_new
import matplotlib.pyplot as plt
class InferencePipeline(object):
    def __init__(
        self,
        model,
        args=None,
        results_path=None,
        args_general=None,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.results_path = results_path
        self.args_general = args_general
        self.image_size = self.args_general.image_size
        self.device = self.args_general.device
        if args_general.inference_method == "SAC":
            self.ppl = build_ppl_new(self.args_general)
        if args_general.inference_method == "MPC":
            self.ppl = build_ppl_new(self.args_general)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
        if not os.path.exists(os.path.join(self.results_path, "thetas")):
            os.makedirs(os.path.join(self.results_path, "thetas"))
        if not os.path.exists(os.path.join(self.results_path, "states")):
            os.makedirs(os.path.join(self.results_path, "states"))
    
    def save(self, sim_id, pred, results_path):
        pred_states, pred_thetas = pred
        for index in range(sim_id.shape[0]):
            id = sim_id[index].cpu().item()
            thetas_filepath = os.path.join(results_path, "thetas", "{}.npy".format(id))
            states_filepath = os.path.join(results_path, "states", "{}.npy".format(id))
            with open(thetas_filepath, 'wb') as file:
                np.save(file, pred_thetas[index].cpu().numpy())
            with open(states_filepath, 'wb') as file:
                np.save(file, pred_states[index].cpu().numpy())  
            print("id {} saved at {}: ".format(id, results_path))
    
    def pad_data(self, state_0, bd_mask_offset_0):
        if state_0.shape[2] != self.image_size:
            assert state_0.shape[1] == 3 and state_0.shape[2] == self.image_size - 2 and state_0.shape[3] == self.image_size - 2
            state_0_pad = torch.zeros(state_0.shape[0], 3, self.image_size, self.image_size)
            state_0_pad[:,:,1:-1,1:-1] = state_0.to(args.device) 
            state_0 = state_0_pad
        if bd_mask_offset_0.shape[2] != self.image_size:
            assert bd_mask_offset_0.shape[1] == 3 and bd_mask_offset_0.shape[2] == self.image_size - 2 and bd_mask_offset_0.shape[3] == self.image_size - 2
            bd_mask_offset_0_pad = torch.zeros(bd_mask_offset_0.shape[0], 3, self.image_size, self.image_size)
            bd_mask_offset_0_pad[:,:,1:-1,1:-1] = bd_mask_offset_0.to(args.device)
            bd_mask_offset_0 = bd_mask_offset_0_pad

        return state_0, bd_mask_offset_0
    
    def MPC_controller_LBFGS(
        self,state_t,
        mask_offsets_t, 
        theta_t,mask_offsets_pad,
        theta_set,num_iters,coef_grad,
        vt_0_to_t=None,
        results_path=None,
        sim_index=None):
        '''
        input: state_t of shape [batch_size, 3, s, s]
                mask_offsets_t of shape [batch_size, 1, s, s]
                theta_t of shape [batch_size, 1]
                time_steps of shape [batch_size, 1]
                mask_offsets_pad of shape [batch_size, num_t, 1, s, s]
                theta_set of shape [batch_size, num_t, 1]
        '''
        args=self.args_general
        state_all = []
        theta_all = []
        force_all = []
        next_state_all = []
        mask_all = []
        updates = 0
        # num_t=20
        num_t=19
        frames=20
        mask_batch = np.ones((1, num_t))
        mask_batch[:, -1] = 0
        theta_all.append(theta_t.detach().cpu().numpy())
        theta_t_init=theta_t
        mask_offsets_t_init=mask_offsets_t
        state_t_init=state_t
        J_list=[]
        force_t_list_total=[]
        force_t_to_T=torch.zeros_like(theta_set)
        vt_t_to_T=torch.zeros_like(theta_set)
        #following revert coef_grad_list
        # coef_grad_list=coef_grad_list[::-1]
        # pdb.set_trace()
        # pdb.set_trace()
        theta_set.requires_grad_(True)
        optimizer = torch.optim.LBFGS([theta_set], line_search_fn='strong_wolfe', lr=coef_grad)
        start = time.time()
        for i in tqdm.trange(num_iters):#update theta_set
            # coef_grad=coef_grad_list[-i]
            pdb.set_trace()
            def closure():
                optimizer.zero_grad()
                theta_t=theta_t_init.detach()
                mask_offsets_t=mask_offsets_t_init.detach()
                state_t=state_t_init.detach()
                J=torch.zeros_like(theta_t)
                J.requires_grad_(True)
                force_t_to_T=torch.zeros_like(theta_set)
                vt_t_to_T=torch.zeros_like(theta_set)
                # vt_0_to_T=torch.zeros((theta_set.shape[0],20))
                # if vt_0_to_t!=None:
                #     vt_0_to_t=vt_0_to_t
                #     vt_0_to_T[:,:(20-theta_set.shape[1])]=vt_0_to_t
                for t in range(theta_set.shape[1]):
                        # theta_t.requires_grad = True
                        # state_t.requires_grad = True
                        # mask_offsets_t.requires_grad = True
                        ###############################
                        # theta_set[:,t]
                        ###############################
                    # pdb.set_trace()
                    # output state and force of next time step
                    
                    if t==0:
                        state_t, force_t = self.ppl.run(state_t, mask_offsets_t, theta_set[:,t]-theta_t)
                        mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_set[:,t]-theta_t) # output mask_offsets of next time step
                    else:
                        state_t, force_t = self.ppl.run(state_t, mask_offsets_t, theta_set[:,t]-theta_set[:,t-1])
                        mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_set[:,t]-theta_set[:,t-1]) # output mask_offsets of next time step
                    if t>0:
                        # pdb.set_trace()
                        force_t_to_T[:,t-1]=force_t
                        vt_t_to_T[:,t-1]=force_t_to_T.sum(-1)
                    # if t != args.frames-1:
                    #     J=J+force_t
                    # else:
                    #     # pdb.set_trace()
                    #     J=J+force_t - 0.1 * ((mask_offsets_t - mask_offsets_pad[:,-1])**2).sum((1,2,3))
                        # pdb.set_trace()
                    force_t_list_total.append(force_t.clone().detach().cpu().numpy()[0])
                force_t = self.ppl.run(state_t, mask_offsets_t)
                force_t_to_T[:,-1]=force_t
                vt_t_to_T[:,-1]=force_t_to_T.sum(-1)
                # vt_0_to_T[:,(20-theta_set.shape[1]):]=vt_t_to_T
                
                gama=0.01
                J=-(vt_t_to_T.mean(-1)-args.lamda*reg_theta(theta=theta_set)-gama * ((mask_offsets_t - mask_offsets_pad[:,0])**2).sum((1,2,3))-0.001*((mask_offsets_t - mask_offsets_pad[:,0]).abs()).sum((1,2,3)))
                # J=vt_0_to_T.mean()-args.lamda*reg_theta(theta=theta_set).sum()-gama * ((mask_offsets_t - mask_offsets_pad[:,-1])**2).sum()
                J_list.append(J.clone().detach().cpu().numpy()[0])
                J.backward()
                return J
            optimizer.step(closure)
        end = time.time()
        print(f"optimize theta {num_iters} iters spend {end-start} s")
        if theta_set.shape[1]%5==0:
            plt.figure(figsize=(20,5))
            plt.plot(J_list)
            plt.savefig(results_path+f"/J_list-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}_gama-{gama}.png")
            plt.show()
            plt.close()
            # plt.figure(figsize=(20,5))
            # plt.plot(force_t_to_T.detach().cpu().numpy()[0])
            # plt.savefig(f"results/force_list-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}.png")
            # plt.show()
            # plt.close()
            
            # ##following plot vt_t_to_T
            
            # plt.figure(figsize=(20,5))
            # plt.plot(vt_t_to_T.detach().cpu().numpy()[0])
            # plt.savefig(f"results/vt-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}.png")
            # plt.show()
            # plt.close()

        if theta_set.shape[1]==num_t:
            theta_list=torch.zeros((1,frames),device=theta_set.device)
            theta_list[:,1:]=theta_set
            theta_list[:,0]=theta_t_init
            #following save theta_set and plot theta_set and save fig
            results_path_SL=results_path+"/SL"
            if not os.path.exists(results_path_SL+"/thetas"):
                os.makedirs(results_path_SL+"/thetas")
            if not os.path.exists(results_path_SL+"/states"):
                os.makedirs(results_path_SL+"/states")
            np.save(results_path_SL+f"/thetas/{sim_index}.npy",theta_list)
            plt.figure()
            plt.plot(theta_list.cpu().detach()[0])
            plt.savefig(results_path_SL+f"/thetas/{sim_index}.png")
            
            plt.figure(figsize=(20,5))
            plt.plot(J_list)
            plt.savefig(results_path_SL+f"/J_list-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}_gama-{gama}.png")
            plt.show()
            plt.close()
        theta_set_star=theta_set
        theta_set_star.requires_grad_(True)  
        # pdb.set_trace()
        # w=comp_work(theta_set_star)    
        return theta_set_star.detach()
    
    def MPC_controller(
        self,state_t,
        mask_offsets_t, 
        theta_t,mask_offsets_pad,
        theta_set,num_iters,coef_grad,
        vt_0_to_t=None,
        results_path=None,
        sim_index=None,theta_cond=None):
        '''
        input: state_t of shape [batch_size, 3, s, s]
                mask_offsets_t of shape [batch_size, 1, s, s]
                theta_t of shape [batch_size, 1]
                time_steps of shape [batch_size, 1]
                mask_offsets_pad of shape [batch_size, num_t, 1, s, s]
                theta_set of shape [batch_size, num_t, 1]
        '''
        args=self.args_general
        state_all = []
        theta_all = []
        force_all = []
        next_state_all = []
        mask_all = []
        updates = 0
        # num_t=20
        num_t=19
        frames=20
        mask_batch = np.ones((1, num_t))
        mask_batch[:, -1] = 0
        theta_all.append(theta_t.detach().cpu().numpy())
        theta_t_init=theta_t
        mask_offsets_t_init=mask_offsets_t
        state_t_init=state_t
        J_list=[]
        force_t_list_total=[]
        force_t_to_T=torch.zeros_like(theta_set)
        vt_t_to_T=torch.zeros_like(theta_set)
        J=torch.zeros_like(theta_t)
        # coef_grad_list=torch.linspace(coef_grad/num_iters, coef_grad, num_iters, dtype = torch.float64)
        coef_grad_list=get_CosineAnnealingLR(num_iters=num_iters,lr_max=coef_grad*(0.85**(num_t-theta_set.shape[1])))
        #following revert coef_grad_list
        # coef_grad_list=coef_grad_list[::-1]
        # pdb.set_trace()
        vt_0_to_T=torch.zeros((theta_set.shape[0],num_t+1))
        # if vt_0_to_t!=None:
        #     vt_0_to_t=vt_0_to_t.detach()
        vt_0_to_T[:,:vt_0_to_t.shape[-1]]=vt_0_to_t
        state_0_T=torch.zeros((state_t.shape[0],num_t+1,state_t.shape[-3],state_t.shape[-1],state_t.shape[-1])).to(state_t.device)
        state_0_T[:,0]=state_t_init.detach()
        for i in tqdm.trange(num_iters):#update theta_set
            # coef_grad=coef_grad_list[-i]
            coef_grad=coef_grad_list[i]
            theta_t=theta_t_init.clone().detach()
            mask_offsets_t=mask_offsets_t_init.clone().detach()
            state_t=state_t_init.clone().detach()
            J=J.clone().detach()*0
            J.requires_grad_(True)
            force_t_to_T=torch.zeros_like(theta_set)
            vt_t_to_T=torch.zeros_like(theta_set)
            vt_0_to_T=vt_0_to_T.detach().to(theta_set.device)
            theta_set=theta_set.detach()
            theta_set.requires_grad_(True)
            for t in range(theta_set.shape[1]):
                
                if t==0:
                    state_t, force_t = self.ppl.run(state_t, mask_offsets_t, theta_set[:,t]-theta_t)
                    mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_set[:,t]-theta_t) # output mask_offsets of next time step
                else:
                    state_t, force_t = self.ppl.run(state_t, mask_offsets_t, theta_set[:,t]-theta_set[:,t-1])
                    mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_set[:,t]-theta_set[:,t-1]) # output mask_offsets of next time step
                state_0_T[:,t+1]=state_t.detach()
                if t>0:
                    # pdb.set_trace()
                    force_t_to_T[:,t-1]=force_t
                    vt_t_to_T[:,t-1]=force_t_to_T.sum(-1)
                # if t != args.frames-1:
                #     J=J+force_t
                # else:
                #     # pdb.set_trace()
                #     J=J+force_t - 0.1 * ((mask_offsets_t - mask_offsets_pad[:,-1])**2).sum((1,2,3))
                    # pdb.set_trace()
                force_t_list_total.append(force_t.clone().detach().cpu().numpy()[0])
            force_t = self.ppl.run(state_t, mask_offsets_t)
            force_t_to_T[:,-1]=force_t
            vt_t_to_T[:,-1]=force_t_to_T.sum(-1)
            # pdb.set_trace()
            # J=-vt_t_to_T.mean(-1)+comp_work(theta_set)
            # J=-vt_t_to_T.mean(-1)
            vt_0_to_T[:,(frames-theta_set.shape[1]):]=vt_t_to_T
            gama=0.01
            # pdb.set_trace()
            # delta_theta_clamp=torch.clamp(theta_set,0.2,1.1)
            # J=vt_0_to_T.mean(-1)-args.lamda*reg_theta(theta=theta_set)-gama * ((mask_offsets_t - mask_offsets_pad[:,0])**2).sum((1,2,3))-args.coef_endcondition*((mask_offsets_t - mask_offsets_pad[:,0]).abs()).sum((1,2,3))
            J=vt_0_to_T.mean(-1)-args.lamda*reg_theta(theta=theta_set)-args.coef_endcondition*(theta_set[:,-1]-theta_cond).abs().sum()-args.coef_clip*(clip_loss(theta_set[:,0]-theta_t_init,-0.15,0.15)+clip_loss(theta_set[:,1:]-theta_set[:,0:theta_set.shape[-1]-1],-0.15,0.15))
            # J=vt_0_to_T.mean()-args.lamda*reg_theta(theta=theta_set).sum()-gama * ((mask_offsets_t - mask_offsets_pad[:,-1])**2).sum()
            # if J[0]>10:
            #     J_list.append(J.clone().detach().cpu().numpy()[0])
            #     force_t_list_total.append(0)
            #     force_t_list_total.append(0)
            #     force_t_list_total.append(0)
            #     break
            # pdb.set_trace()
            grad_J=grad(J, theta_set)[0]
            # pdb.set_trace()
            theta_set=theta_set+coef_grad*grad_J
            theta_set=torch.clamp(theta_set,0.2,1.1)
            J_list.append(J.clone().detach().cpu().numpy()[0])
            
        if theta_set.shape[1]%5==0:
            plt.figure(figsize=(20,5))
            plt.plot(J_list)
            plt.savefig(results_path+f"/J_list-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}_gama-{gama}.png")
            plt.show()
            plt.close()
            
            theta_list=torch.zeros((1,theta_set.shape[-1]+1),device=theta_set.device)
            theta_list[:,1:]=theta_set.detach()
            theta_list[:,0]=theta_t_init
            
            plt.figure(figsize=(20,5))
            plt.plot(theta_list.cpu().detach()[0])
            plt.savefig(results_path+f"/theta-timesteps-{20-theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}_gama-{gama}.png")
            plt.show()
            plt.close()
            plt.figure(figsize=(20,5))
            plt.plot(force_t_to_T.detach().cpu().numpy()[0])
            plt.savefig(results_path+f"/force_list-timesteps-{20-theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}.png")
            plt.show()
            plt.close()
            
            # ##following plot vt_t_to_T
            
            # plt.figure(figsize=(20,5))
            # plt.plot(vt_t_to_T.detach().cpu().numpy()[0])
            # plt.savefig(f"results/vt-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}.png")
            # plt.show()
            # plt.close()

        if theta_set.shape[1]==num_t:
            theta_list=torch.zeros((1,frames),device=theta_set.device)
            theta_list[:,1:]=theta_set
            theta_list[:,0]=theta_t_init
            #following save theta_set and plot theta_set and save fig
            results_path_SL=results_path+"/SL"
            if not os.path.exists(results_path_SL+"/thetas"):
                theta_list_batch=theta_list.detach()
                state_list_batch=state_0_T.detach()
            else:
                theta_list_batch=np.load(results_path_SL+f"/thetas/theta_list_batch.npy")
                state_list_batch=np.load(results_path_SL+f"/states/stata_list_batch.npy")
                theta_list_batch=torch.tensor(theta_list_batch).to(theta_cond.device)
                state_list_batch=torch.tensor(state_list_batch).to(theta_cond.device)
                theta_list_batch=torch.cat((theta_list_batch,theta_list),dim=0)
                state_list_batch=torch.cat((state_list_batch,state_0_T),dim=0)
            if not os.path.exists(results_path_SL+"/thetas"):
                os.makedirs(results_path_SL+"/thetas")
            if not os.path.exists(results_path_SL+"/states"):
                os.makedirs(results_path_SL+"/states")
            
            np.save(results_path_SL+f"/thetas/theta_list_batch.npy",theta_list_batch.detach().cpu().numpy())
            np.save(results_path_SL+f"/states/stata_list_batch.npy",state_list_batch.detach().cpu().numpy())
            # pdb.set_trace()
            plt.figure()
            plt.plot(theta_list.cpu().detach()[0])
            plt.savefig(results_path_SL+f"/thetas/{sim_index}.png")
            
            plt.figure(figsize=(20,5))
            plt.plot(J_list)
            plt.savefig(results_path_SL+f"/J_list-timesteps-{theta_set.shape[1]}_num_iters-{num_iters}_lambda-{args.lamda}_gama-{gama}.png")
            plt.show()
            plt.close()
        theta_set_star=theta_set
        theta_set_star.requires_grad_(True)  
        # pdb.set_trace()
        # w=comp_work(theta_set_star)    
        return theta_set_star.detach()
    def run_model_DDPM(self, state_0, bd_0, thetas_0):
        return self.model.sample(
            design_fn=self.args["design_fn"],
            design_guidance=self.args["design_guidance"],
            cond=[state_0, bd_0], # initial state and bd mask, offsets
            thetas_0=thetas_0, # initial angle theta
            bd_updater=self.args["bd_updater"]
        )

    def run_model_SAC(self, state_0, bd_0, thetas_0):
        s = state_0.shape[-1]
        state_0 = state_0.to(self.device)
        bd_0 = bd_0.to(self.device)
        thetas_0 = thetas_0.to(self.device)
        state_t = state_0
        mask_offsets_t = bd_0
        theta_t = thetas_0
        theta_t_list=torch.zeros(state_0.shape[0], 20).cuda()
        theta_t_list[:, 0] = theta_t
        state_t_list = torch.zeros(state_0.shape[0], 20, 3, s, s).cuda()
        state_t_list[:, 0] = state_t
        for t in range(19):
            state = torch.cat((state_t, mask_offsets_t), dim=1)
            state = torch.cat((state, bd_0), 1)
            theta_current = theta_t
            state_time = torch.cat((normalize_batch(state)[:,:6], theta_current.reshape(-1,1,1,1).repeat(1,1,s,s), \
                                    thetas_0.reshape(-1,1,1,1).repeat(1,1,s,s), t*torch.ones_like(state[:,[0]])), dim=1)
            theta_t = torch.tensor(self.model.select_action(state_time.cpu(), eval=True), device=self.device) # theta1;...;theta20
            theta_t_list[:, t+1] = theta_t
            state_t, force_t = self.ppl.run(state_t, mask_offsets_t, theta_t-theta_current) # output state of next time step and force of current step, s1, f0;...;s20, f19
            mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_t-theta_current) # output mask_offsets of next time step, bd1;...;bd20
            state_t_list[:, t+1] = state_t
        
        return state_t_list, theta_t_list
    
    def run_model_SAC_pob(self, state_0, bd_0, thetas_0):
        s = state_0.shape[-1]
        state_0 = state_0.to(self.device)
        bd_0 = bd_0.to(self.device)
        thetas_0 = thetas_0.to(self.device)
        state_t = torch.zeros(state_0.shape[0], 3, s, s).to(self.device)
        state_t[:, [-1]] = state_0[:, [-1]]
        mask_offsets_t = bd_0
        theta_t = thetas_0
        theta_t_list=torch.zeros(state_0.shape[0], 20).cuda()
        theta_t_list[:, 0] = theta_t
        state_t_list = torch.zeros(state_0.shape[0], 20, 3, s, s).cuda()
        state_t_list[:, 0] = state_t
        for t in range(19):
            state = torch.cat((state_t, mask_offsets_t), dim=1)
            state = torch.cat((state, bd_0), 1)
            theta_current = theta_t
            state_time = torch.cat((normalize_batch(state)[:,2:6], theta_current.reshape(-1,1,1,1).repeat(1,1,s,s), \
                                    thetas_0.reshape(-1,1,1,1).repeat(1,1,s,s), t*torch.ones_like(state[:,[0]])), dim=1)
            theta_t = torch.tensor(self.model.select_action(state_time.cpu(), eval=True), device=self.device) # theta1;...;theta20
            theta_t_list[:, t+1] = theta_t
            state_t_, force_t = self.ppl.run_vis_pressure(state_t[:,[-1]], mask_offsets_t, theta_t-theta_current) # output state of next time step and force of current step, s1, f0;...;s20, f19
            state_t[:, [-1]] = state_t_
            mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_t-theta_current) # output mask_offsets of next time step, bd1;...;bd20
            state_t_list[:, t+1] = state_t
        
        return state_t_list, theta_t_list
    
    def run_model_MPC(self, state_0, bd_0, thetas_0):
        '''
        state_0 [batch_size,3,64,64]
        bd_0    [batch_size,3,64,64]
        thetas_0 [batch_size]
        '''
        # pdb.set_trace()
        # num_t=20
        frames=20
        num_t=19
        args=self.args_general
        if args.only_vis_pressure:
            state_0=state_0[:,[-1],:,:]
        state_0_batch=state_0.detach()
        bd_0_batch=bd_0.detach()
        thetas_0_batch=thetas_0.detach()
        theta_t_list_batch=torch.zeros(state_0_batch.shape[0], frames).cuda()
        s = state_0.shape[-1]
        # pdb.set_trace()
        state_t_list_batch = torch.zeros(state_0_batch.shape[0], frames, state_0.shape[-3], s, s).cuda()
        for i_temp in range(state_0.shape[0]):
            state_0 = state_0_batch[[i_temp]].to(self.device)
            bd_0 = bd_0_batch[[i_temp]].to(self.device)
            thetas_0 = thetas_0_batch[[i_temp]].to(self.device)
            state_t = state_0
            mask_offsets_t = bd_0
            theta_t = thetas_0
            theta_t_list=torch.zeros(state_0.shape[0], frames).cuda()
            theta_t_list[:, 0] = theta_t
            theta_cond=theta_t.detach()
            state_t_list = torch.zeros(state_0.shape[0], frames, state_0.shape[-3], s, s).cuda()
            state_t_list[:, 0] = state_t
            

            theta_set= 0.7* torch.rand((1,num_t)) + 0.2
            # theta_set= 0.7* torch.rand((1,20)) + 0.6
            # theta_set= theta_t.reshape(theta_t.shape[0],-1).clone().repeat(1,20)
            # theta_set=theta_set_preset[:,1:] #[1,20]
            theta_set=theta_set.to(self.device)
            theta_set.requires_grad_(True)
            force_t_to_T=torch.zeros((theta_t.shape[0],frames)).to(self.device)
            vt_t_to_T=torch.zeros((theta_t.shape[0],frames)).to(self.device)
            v_0_to_t=[]
            for t in range(num_t):
                state = torch.cat((state_t, mask_offsets_t), dim=1)
                state = torch.cat((state, bd_0), 1)
                theta_current = theta_t.detach()
                ###############
                force_t= self.ppl.run(state_t, mask_offsets_t)
                force_t_to_T[:,t]=force_t.clone().detach()
                vt_t_to_T[:,t]=force_t_to_T.sum(-1)
                if t==0:
                # pass
                    # theta_set=self.MPC_controller_LBFGS(state_t,mask_offsets_t= mask_offsets_t, theta_t=theta_t,mask_offsets_pad=bd_0,theta_set=theta_set.detach(),num_iters=args.num_iters,coef_grad=args.coef_grad,results_path=args.inference_result_subpath,sim_index=i_temp).detach()
                    theta_set=self.MPC_controller(state_t,mask_offsets_t= mask_offsets_t, theta_t=theta_t,mask_offsets_pad=bd_0,theta_set=theta_set.detach(),num_iters=args.num_iters,coef_grad=args.coef_grad,vt_0_to_t=vt_t_to_T[:,:t+1].detach(),results_path=args.inference_result_subpath,sim_index=i_temp,theta_cond=theta_cond).detach()
                else:
                    # pdb.set_trace()
                    state_t.requires_grad_(True)
                    mask_offsets_t.requires_grad_(True)
                    theta_set.requires_grad_(True)
                    
                    # theta_set=self.MPC_controller_LBFGS(state_t, mask_offsets_t=mask_offsets_t,theta_t= theta_t,mask_offsets_pad=bd_0,theta_set=theta_set.detach(),num_iters=args.num_iters,coef_grad=args.coef_grad,vt_0_to_t=vt_t_to_T[:,:t].detach(),results_path=args.inference_result_subpath).detach()
                    theta_set=self.MPC_controller(state_t, mask_offsets_t=mask_offsets_t,theta_t= theta_t,mask_offsets_pad=bd_0,theta_set=theta_set.detach(),num_iters=args.num_iters,coef_grad=args.coef_grad,vt_0_to_t=vt_t_to_T[:,:t+1].detach(),results_path=args.inference_result_subpath,theta_cond=theta_cond).detach()
                ###############
                with torch.no_grad():
                    state_t, _ = self.ppl.run(state_t, mask_offsets_t, theta_set[:,0]-theta_t) # output state of next time step and force of current step, s1, f0;...;s20, f19
                    mask_offsets_t = self.ppl.update_mask_offsets_new(mask_offsets_t, theta_set[:,0]-theta_t) # output mask_offsets of next time step, bd1;...;bd20
                theta_t=theta_set[:,0].detach()
                theta_t_list[:, t+1] = theta_t
                if t != num_t-1:
                    theta_set=theta_set[:,1:].detach()
                state_t_list[:, t+1] = state_t
            theta_t_list_batch[[i_temp]]=theta_t_list
            state_t_list_batch[[i_temp]]=state_t_list

            #then chose theta_t_list[:,19] to plt and save png
            plt.figure()
            plt.plot(theta_t_list[0].cpu().numpy())
            plt.savefig('theta_t_list_??????.png') 
        return state_t_list_batch.detach(), theta_t_list_batch.detach() #[batch_size,3,64,64] [batch_size,21]

    def run(self, dataloader):
        for i, data in enumerate(dataloader):
            # pdb.set_trace()
            state_0, thetas_0, bd_mask_offset_0, sim_id, _ = data
            state_0, bd_0 = self.pad_data(state_0, bd_mask_offset_0)
            if self.args_general.inference_method == "SAC":
                if self.args_general.only_vis_pressure:
                    pred = self.run_model_SAC_pob(state_0, bd_0, thetas_0)
                else:
                    pred = self.run_model_SAC(state_0, bd_0, thetas_0)
            elif self.args_general.inference_method == "DDPM":
                pred = self.run_model_DDPM(state_0, bd_0, thetas_0)
            elif self.args_general.inference_method == "MPC":
                pred = self.run_model_MPC(state_0, bd_0, thetas_0)
            self.save(sim_id, pred, self.results_path) # pred should contain pred_thetas and pred_states


def inference(dataloader, diffusion, bd_updater, design_fn, args):
    model = diffusion # may vary according to different control methods
    model_args = {
        "design_fn": design_fn,
        "design_guidance": args.design_guidance,
        "bd_updater": bd_updater
    } # may vary according to different control methods

    inferencePPL = InferencePipeline(
        model, 
        model_args,
        results_path = args.inference_result_subpath,
        args_general=args
    )
    inferencePPL.run(dataloader)


def load_data(args):
    dataset = Jellyfish(
        dataset="jellyfish", 
        dataset_path=args.dataset_path,
        time_steps=40, 
        steps=args.frames, 
        time_interval=1, 
        is_train=False, 
        is_testdata=args.is_testdata,
        only_vis_pressure=args.only_vis_pressure
    )
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    print("number of batch in test_loader: ", len(test_loader))
    # pdb.se
    return test_loader

def main(args):
    force_model, diffusion, bd_updater, design_fn = load_model(args)
    dataloader = load_data(args)
    inference(dataloader, diffusion, bd_updater, design_fn, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference 2d inverse design model')
    parser.add_argument('--dataset', default='jellyfish', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--dataset_path', default=JELLYFISH_DATA_PATH, type=str,
                        help='path to dataset')
    parser.add_argument('--only_vis_pressure', action='store_true', 
                        help="whether only observe pressure, only used in simulator model")
    parser.add_argument('--use_guidance_in_model_predictions', action='store_true', 
                        help="whether use guidance in model_predictions function")
    parser.add_argument('--batch_size', default=10, type=int,
                        help='size of batch of input to use')
    parser.add_argument('--num_batches', default=20, type=int,
                        help='num of batches to evaluate')
    parser.add_argument('--frames', default=20, type=int,
                        help='number of time steps of states')
    parser.add_argument('--backward_steps', default=5, type=int,
                        help='number of backward_steps for universal-backward sampling')
    parser.add_argument('--backward_lr', default=0.01, type=float,
                        help='backward_lr for universal-backward sampling')
    parser.add_argument('--standard_fixed_ratio', default=0.003, type=float,
                        help='standard_fixed_ratio for standard sampling')
    parser.add_argument('--forward_fixed_ratio', default=0.01, type=float,
                        help='forward_fixed_ratio for universal-forward sampling')
    parser.add_argument('--coeff_ratio', default=0.1, type=float,
                        help='coeff_ratio for standard-alpha sampling')
    parser.add_argument('--coeff_ratio_J', default=0.3, type=float,
                        help='coeff_ratio of gradient of objective for standard-alpha sampling')
    parser.add_argument('--coeff_ratio_w', default=0.3, type=float,
                        help='coeff_ratio of predicted noise of p(w) for standard-alpha sampling')
    parser.add_argument('--reg_ratio', default=1000, type=float,
                        help='tradeoff hyperparamter for objective and regularization')
    parser.add_argument('--cond_steps', default=1, type=int, help='number of steps to condition on')
    parser.add_argument('--diffusion_joint_model_path', default=os.path.join(JELLYFISH_DATA_PATH, "checkpoints"), 
                        type=str,
                        help='directory of trained diffusion model of the joint distribution of u and w')
    parser.add_argument('--diffusion_w_model_path', default=os.path.join(JELLYFISH_DATA_PATH, "checkpoints"), type=str,
                        help='directory of trained diffusion thetas model (Unet)')
    parser.add_argument('--diffusion_joint_checkpoint', default=100, type=int,
                        help='index of checkpoint of trained diffusion model of the joint distribution of u and w')
    parser.add_argument('--diffusion_w_checkpoint', default=50, type=int,
                        help='index of checkpoint of trained diffusion thetas model (Unet)')
    parser.add_argument('--sampling_timesteps', default=1000, type=int,
                        help='diffusion steps of ddim, if 1000, ddpm')
    
    # SAC model
    parser.add_argument('--sac_model_path', default="/home/user/pde_gen_control-main/results/2d_pde_sac/SAC_policy_track2023-12-06 10:59:41.842475_0_99", type=str,
                        help='directory of trained sac model')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--is_testdata', default=True, type=bool,
                    help='whether run mini example data, if True, yes; otherwise, run full data')
    parser.add_argument('--inference_result_path', default="/user/project/pde_gen_control/results_0106/", type=str,
                        help='path to save inference result')
    parser.add_argument('--inference_result_subpath', default="/user/project/pde_gen_control/results_0106/", type=str,
                        help='subpath to save inference result')
    parser.add_argument('--log_path', default="/user/pde_gen_control/results/", type=str,
                    help='folder to save training logs')
    parser.add_argument('--design_guidance', default='standard-alpha', type=str,
                        help='design_guidance')
    parser.add_argument('--inference_method', default="DDPM", type=str,
                        help='the inference method: DDPM | SAC | MPC')
    parser.add_argument('--force_model_checkpoint', type=str, default=os.path.join(JELLYFISH_DATA_PATH, "checkpoints/force_surrogate_model/force_model_epoch_9.pth"))
    # parser.add_argument('--simulator_model_checkpoint', type=str, default="/user/project/pde_gen_control/checkpoints/epoch_9_no_mask.pth")
    parser.add_argument('--boundary_updater_model_checkpoint', type=str, default=os.path.join(JELLYFISH_DATA_PATH, "checkpoints/boundary_updater/boundary_updater_epoch_9.pth"))
    parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
    parser.add_argument('--num_iters', type=int, default=1, metavar='N',
                help='number of optimization iters')
    parser.add_argument('--coef_grad', type=float, default=0.0001, metavar='N',
                    help='coef of gradient of scalar J with respect to vector x')
    parser.add_argument('--coef_endcondition', type=float, default=0.0002, metavar='N',
                    help='coef of end condition')
    parser.add_argument('--lamda', type=float, default=100, metavar='N',
                    help='lambda of reg_theta(theta)')

    parser.add_argument('--coef_clip', type=float, default=0, metavar='N',
                    help='coef of clip loss')
    parser.add_argument('--w_prob_exp', type=float, default=0.7, metavar='N',
                    help='gamma of prior reweighting in the paper, should be within (0,1], if 1.0 means no reweighting, i.e., DiffPhyCon-lite')
    # get_ipython().run_line_magic('matplotlib', 'inline')

    args = parser.parse_args()
    args.device = torch.device('cuda', args.gpu)
    if args.only_vis_pressure:
        args.diffusion_joint_model_path = os.path.join(JELLYFISH_DATA_PATH, "checkpoints", "joint_partial")
        args.diffusion_w_model_path = os.path.join(JELLYFISH_DATA_PATH, "checkpoints", "w_partial")
        args.inference_result_path = os.path.join(JELLYFISH_DATA_PATH, "results", "inference_partial")
        args.log_path = os.path.join(JELLYFISH_DATA_PATH, "logs", "inference_partial")
    else:
        args.diffusion_joint_model_path = os.path.join(JELLYFISH_DATA_PATH, "checkpoints", "joint_full")
        args.diffusion_w_model_path = os.path.join(JELLYFISH_DATA_PATH, "checkpoints", "w_full")
        args.inference_result_path = os.path.join(JELLYFISH_DATA_PATH, "results", "inference_full")
        args.log_path = os.path.join(JELLYFISH_DATA_PATH, "logs", "inference_full")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.inference_result_subpath = os.path.join(
        args.inference_result_path,
        current_time +  "_coeff_ratio_w_{}_J{}_".format(args.coeff_ratio_w, args.coeff_ratio_J)
    )
    print("args: ", args)
    main(args)