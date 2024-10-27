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
            data = np.load(os.path.join(self.online_dir, 'memory_{}.npz'.format(index)))
        except:
            data = np.load(os.path.join(self.offline_dir, 'memory_{}.npz'.format(index)))
            
        now_state_all = data['now_state_all']
        action_all = data['action_all']
        next_state_all = data['next_state_all']
        mask_batch_all = data['mask_batch_all']
        
        now_state_all[:,-1] = now_state_all[:,-1]/31
        next_state_all[:,-1] = next_state_all[:,-1]/31
        
        return now_state_all, action_all, next_state_all, mask_batch_all
    
    def __len__(self):
        return len(os.listdir(self.offline_dir)) - 1
    