import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import random
import os

class SAC_Jellyfish(Dataset):
    def __init__(self, 
                vx_min,
                vx_max,
                vy_min,
                vy_max,
                p_min,
                p_max,
                offline_dir = "data/memory_2d/", 
                online_dir = "data/memory_2d/online/",
                ):
        self.offline_dir = offline_dir
        self.online_dir = online_dir
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.vy_min = vy_min
        self.vy_max = vy_max
        self.p_min = p_min
        self.p_max = p_max
    
    def normalize(self, state_full):
        state_full[0] = ((state_full[0,:,:] - self.vx_min) / (self.vx_max - self.vx_min) - 0.5) * 2
        state_full[1] = ((state_full[1,:,:] - self.vy_min) / (self.vy_max - self.vy_min) - 0.5) * 2
        state_full[2] = ((state_full[2,:,:] - self.p_min) / (self.p_max - self.p_min) - 0.5) * 2
        return state_full
    
    def normalize_batch(self, state_full):
        state_full[:,0] = ((state_full[:,0,:,:] - self.vx_min) / (self.vx_max - self.vx_min) - 0.5) * 2
        state_full[:,1] = ((state_full[:,1,:,:] - self.vy_min) / (self.vy_max - self.vy_min) - 0.5) * 2
        state_full[:,2] = ((state_full[:,2,:,:] - self.p_min) / (self.p_max - self.p_min) - 0.5) * 2
        return state_full

    def __getitem__(self, index):
        try:
            data = np.load(os.path.join(self.online_dir, '{}.npz'.format(index)))
        except:
            data = np.load(os.path.join(self.offline_dir, '{}.npz'.format(index)))
                    
        cond_theta_all = data['cond_theta_all']
        now_state_all = data['now_state_all']
        time_all = data['time_all']
        thetas_all = data['thetas_all']
        thetas_delta_all = data['thetas_delta_all']
        reg_all = thetas_delta_all**2
        force_all = data['force_all']
        cond_T_all = (thetas_all - cond_theta_all)**2 + 0.1*np.abs(thetas_all - cond_theta_all)
        next_state_all = data['next_state_all']
        mask_batch_all = data['mask_batch_all']

        now_state_all = self.normalize(torch.tensor(now_state_all))
        next_state_all = self.normalize(torch.tensor(next_state_all))
        now_state_all = np.concatenate((now_state_all[:6], (thetas_all-thetas_delta_all)*np.ones_like(now_state_all[[0]]), cond_theta_all*np.ones_like(now_state_all[[0]]), time_all*np.ones_like(now_state_all[[0]])), axis=0)
        next_state_all = np.concatenate((next_state_all[:6], thetas_all*np.ones_like(now_state_all[[0]]), cond_theta_all*np.ones_like(now_state_all[[0]]), (time_all+1)*np.ones_like(now_state_all[[0]])), axis=0)
        return now_state_all, thetas_all, force_all, cond_T_all, reg_all, next_state_all, mask_batch_all
    
    def __len__(self):
        return len(os.listdir(self.offline_dir)) - 1


class SAC_Jellyfish_pob(Dataset):
    def __init__(self, 
                vx_min,
                vx_max,
                vy_min,
                vy_max,
                p_min,
                p_max,
                offline_dir = "data/memory_2d/", 
                online_dir = "data/memory_2d/online/",
                ):
        self.offline_dir = offline_dir
        self.online_dir = online_dir
        self.vx_min = vx_min
        self.vx_max = vx_max
        self.vy_min = vy_min
        self.vy_max = vy_max
        self.p_min = p_min
        self.p_max = p_max
    
    def normalize(self, state_full):
        state_full[0] = ((state_full[0,:,:] - self.vx_min) / (self.vx_max - self.vx_min) - 0.5) * 2
        state_full[1] = ((state_full[1,:,:] - self.vy_min) / (self.vy_max - self.vy_min) - 0.5) * 2
        state_full[2] = ((state_full[2,:,:] - self.p_min) / (self.p_max - self.p_min) - 0.5) * 2
        return state_full
    
    def normalize_batch(self, state_full):
        state_full[:,0] = ((state_full[:,0,:,:] - self.vx_min) / (self.vx_max - self.vx_min) - 0.5) * 2
        state_full[:,1] = ((state_full[:,1,:,:] - self.vy_min) / (self.vy_max - self.vy_min) - 0.5) * 2
        state_full[:,2] = ((state_full[:,2,:,:] - self.p_min) / (self.p_max - self.p_min) - 0.5) * 2
        return state_full

    def __getitem__(self, index):
        try:
            data = np.load(os.path.join(self.online_dir, '{}.npz'.format(index)))
        except:
            data = np.load(os.path.join(self.offline_dir, '{}.npz'.format(index)))
            
        
        cond_theta_all = data['cond_theta_all']
        now_state_all = data['now_state_all']
        time_all = data['time_all']
        thetas_all = data['thetas_all']
        thetas_delta_all = data['thetas_delta_all']
        reg_all = thetas_delta_all**2
        force_all = data['force_all']
        cond_T_all = (thetas_all - cond_theta_all)**2 + 0.1*np.abs(thetas_all - cond_theta_all)
        next_state_all = data['next_state_all']
        mask_batch_all = data['mask_batch_all']

        now_state_all = self.normalize(torch.tensor(now_state_all))
        next_state_all = self.normalize(torch.tensor(next_state_all))
        now_state_all = np.concatenate((now_state_all[2:6], (thetas_all-thetas_delta_all)*np.ones_like(now_state_all[[0]]), cond_theta_all*np.ones_like(now_state_all[[0]]), time_all*np.ones_like(now_state_all[[0]])), axis=0)
        next_state_all = np.concatenate((next_state_all[2:6], thetas_all*np.ones_like(now_state_all[[0]]), cond_theta_all*np.ones_like(now_state_all[[0]]), (time_all+1)*np.ones_like(now_state_all[[0]])), axis=0)

        return now_state_all, thetas_all, force_all, cond_T_all, reg_all, next_state_all, mask_batch_all
    
    def __len__(self):
        return len(os.listdir(self.offline_dir)) - 1
    