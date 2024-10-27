import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.sparse import load_npz
import pdb
import sys, os
import math
import random
from ddpm.utils import p, cycle

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))


class SimulatorData(Dataset):
    def __init__(
        self,
        dataset_path,
        size = 64,
        time_steps=256,
        steps=32,
        is_train=True,
    ):
        super().__init__()
        self.root = dataset_path
        self.size = size
        self.steps = steps
        self.time_steps = time_steps
        self.time_interval = int(time_steps/steps)
        self.time_stamps = 31
        self.init_velocity = torch.tensor(np.load(os.path.join(self.root, 'intial_vel.npy')), dtype=torch.float)\
                            .permute(2, 0, 1).reshape(2, 1, 128, 128)
        self.is_train = is_train
        self.dirname = "train" if self.is_train else "test"
        self.sub_dirname = '32_128_128_testdata'
        if self.is_train:
            self.n_simu = 20000 * self.time_stamps
        else:
            self.n_simu = 200 * self.time_stamps
        self.RESCALER = torch.tensor([3, 20, 20, 17, 19, 1]).reshape(6, 1, 1, 1) 

    def __len__(self):
        return self.n_simu

    def locate_index(self, idx):
        sim_id, time_id = divmod(idx, self.time_stamps)
        assert 0 <= sim_id < self.n_simu
        return sim_id, time_id
    
    def __getitem__(self, idx):
        sim_id, time_id = self.locate_index(idx)  
        if self.is_train:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.cat((self.init_velocity[:, :, ::2, ::2], v), dim=1)[:, :33]
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float)
            s = s[:, 1]/s.sum(-1)
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size)
            state = torch.cat((d, v, c, s), dim=0)[:, :32] / self.RESCALER # 6, 32, 64, 64, rescaled!
        
            data = (
                state[:, time_id], # density, velocity, control, smoke_portion
                torch.cat((state[:3], state[[-1]]), dim=0)[:, time_id + 1], # density, velocity, smoke_portion
                time_id
            )

        else:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.cat((self.init_velocity, v), dim=1)[:, :33]
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float)
            s = s[:, 1]/s.sum(-1)
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], 128, 128)
            state = torch.cat((d, v, c, s), dim=0)[:, :32, ::2, ::2] # 6, 32, 64, 64, not rescaled!
        
            data = (
                state[:, time_id], # density, velocity, control, smoke_portion
                torch.cat((state[:3], state[[-1]]), dim=0)[:, time_id + 1], # density, velocity, smoke_portion
                time_id
            )
        return data


if __name__ == "__main__":
    dataset = SimulatorData(
        dataset_path="/data/2d/",
        is_train=True,
    )
    print(len(dataset))
    data = dataset[4]
    print(data[0].shape, data[1], data[2], data[3])
