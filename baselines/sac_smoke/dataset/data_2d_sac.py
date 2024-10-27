import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import random
import os
    

class SAC_Smoke(Dataset):
    def __init__(self, 
                RESCALER = 1,
                num_t = 32,
                offline_dir = "/data/2d/rl/", 
                online_dir = "/data/2d/rl/online/",
                ):
        self.offline_dir = offline_dir
        self.online_dir = online_dir
        self.RESCALER = RESCALER
        self.num_t = num_t
    
    def normalize(self, state_full):
        state_full[-1] = state_full[-1] / self.num_t # num_t
        return state_full
    
    def normalize_batch(self, state_full):
        state_full[:,-1] = state_full[:,-1] / self.num_t # num_t
        return state_full

    
    def __getitem__(self, index):
        try:
            data = np.load(os.path.join(self.online_dir, 'memory_{}.npz'.format(index)))
        except:
            data = np.load(os.path.join(self.offline_dir, 'memory_{}.npz'.format(index)))
        now_state_all = data['now_state_all']
        now_state_all = np.concatenate((now_state_all[:3], now_state_all[[-1]]), 0)

        action_all = data['action_all']

        next_state_all = data['next_state_all']
        reward_batch = np.mean(next_state_all[-2])
        next_state_all = np.concatenate((next_state_all[:3], next_state_all[[-1]]), 0)

        mask_batch_all = data['mask_batch_all'][0,0,0]

        now_state_all = self.normalize(torch.tensor(now_state_all))
        next_state_all = self.normalize(torch.tensor(next_state_all))
        # print(now_state_all.shape, action_all.shape, next_state_all.shape, reward_batch.shape, mask_batch_all.shape)
        # print(now_state_all.max(), action_all.max(), next_state_all.max(), reward_batch.max(), mask_batch_all.max())

        return now_state_all, action_all, next_state_all, reward_batch, mask_batch_all
    
    def __len__(self):
        return len(os.listdir(self.offline_dir)) - 1

