import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import pdb
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import random

class Jellyfish(Dataset):
    def __init__(
        self,
        dataset,
        dataset_path,
        time_steps=40,
        steps=20,
        time_interval=1,
        is_train=True,
        is_testdata = False,
        for_pipeline = False,
        only_vis_pressure=False,
    ):
        super().__init__()
        self.dataset = dataset
        if not self.dataset.startswith('jellyfish'):
            raise
        self.root = dataset_path
        self.steps = steps
        self.time_steps = time_steps
        self.time_interval = time_interval
        self.is_train = is_train
        self.is_testdata = is_testdata
        self.win_size = self.steps * self.time_interval
        self.for_pipeline = for_pipeline
        self.only_vis_pressure = only_vis_pressure
        self.dirname = "train_data" if self.is_train else "test_data"

        if self.is_testdata:
            self.n_simu = 100 if self.is_train else 50
        else:
            self.n_simu = 1000 if self.is_train else 100 
        self.time_steps_effective = (self.time_steps - self.win_size) // self.time_interval
        
        normalization_filename = os.path.join(self.root, self.dirname, "normalization_max_min.pkl")
        if os.path.isfile(normalization_filename): 
            normdict = pickle.load(open(normalization_filename, "rb"))
            self.vx_max = normdict["vx_max"]
            self.vx_min = normdict["vx_min"]
            self.vy_max = normdict["vy_max"]
            self.vy_min = normdict["vy_min"]
            self.p_max = normdict["p_max"]
            self.p_min = normdict["p_min"]
            print("self.vx_max, self.vx_min, self.vy_max, self.vy_min, self.p_max, self.p_min: ", self.vx_max, self.vx_min, self.vy_max, self.vy_min, self.p_max, self.p_min)
        else:
            raise 

    def __len__(self):
        if self.is_train:
            return self.n_simu * self.time_steps_effective
        else:
            return self.n_simu

    def __getitem__(self, idx):
        if self.for_pipeline or self.is_train:
            sim_id, time_id = divmod(idx, self.time_steps_effective)
        else:
            sim_id, time_id = idx, 0 # for test, pass each trajectory as a whole and only once
        state_full = torch.FloatTensor(np.load(os.path.join(self.root, self.dirname, "states","sim_{:06d}.npz".format(sim_id)))["a"]) # [40, 3, 64, 64]
        if not self.for_pipeline:
            pressure = (torch.clamp((state_full[:,2,:,:] - self.p_min) / (self.p_max - self.p_min), 0, 1) - 0.5).unsqueeze(1) * 2
            if not self.only_vis_pressure:
                vx = (torch.clamp((state_full[:,0,:,:] - self.vx_min) / (self.vx_max - self.vx_min), 0, 1) - 0.5).unsqueeze(1) * 2
                vy = (torch.clamp((state_full[:,1,:,:] - self.vy_min) / (self.vy_max - self.vy_min), 0, 1) - 0.5).unsqueeze(1) * 2
                state_full = torch.cat((vx, vy, pressure), 1)
            else:
                state_full = pressure
        state_full[torch.isnan(state_full)] = 0 # [40, 3, 64, 64]
    
        state = state_full[time_id:time_id + self.win_size] # [20, 3, 64, 64]
        bd_mask_offset_full = np.load(os.path.join(self.root, self.dirname, 'bdry_merged_mask_offsets/sim_{:06d}.npz'.format(sim_id)))["a"] # [40, 62, 62, 3]
        bd_mask_offset = torch.FloatTensor(
            np.transpose(
                bd_mask_offset_full[time_id:time_id + self.win_size],
                (0, 3, 1, 2)
            )
        ) # [20, 3, 62, 62]
        bd_mask_offset[torch.isnan(bd_mask_offset)] = 0
        
        thetas_full = np.load(os.path.join(self.root, self.dirname, 'bdry_head_thetas/sim_{:06d}.npz'.format(sim_id)))["thetas"] # [40]
        thetas = torch.FloatTensor(
            thetas_full[time_id:time_id + self.win_size]
        )
        
        if not self.for_pipeline:
            if self.is_train: # train diffusion model
                data = (
                    state,
                    bd_mask_offset,
                    thetas,
                    sim_id,
                    time_id,
                )
            else: # test diffusion model
                state_0 = state_full[0]
                bd_mask_offset_0 = torch.FloatTensor(
                    np.transpose(
                        bd_mask_offset_full[0,:,:,:],
                        (2, 0, 1)
                    )
                )
                thetas_0 = thetas[0] 
                thetas_gt = torch.FloatTensor(
                    thetas_full[:self.win_size]
                )
                data = (
                    state_0,
                    thetas_0,
                    bd_mask_offset_0, # boundary mask and offset of initial time step of each trajectory
                    sim_id,
                    thetas_gt
                )
        else:
            bd_mask_offset_0 = torch.FloatTensor(
                np.transpose(
                    bd_mask_offset_full[0,:,:,:],
                    (2, 0, 1)
                )
            ) # [3, 126, 126]

            data = (
                state,
                bd_mask_offset,
                thetas,
                bd_mask_offset_0, # boundary mask and offset of initial time step of each trajectory
                sim_id,
                time_id
            )
    
        return data

class Smoke(Dataset):
    def __init__(
        self,
        dataset_path,
        time_steps=256,
        steps=32,
        all_size=128,
        size=64,
        is_train=True,
    ):
        super().__init__()
        self.root = dataset_path
        self.steps = steps
        self.time_steps = time_steps
        self.time_interval = int(time_steps/steps)
        self.all_size = all_size
        self.size = size
        self.space_interval = int(all_size/size)
        self.is_train = is_train
        self.dirname = "train" if self.is_train else "test"
        self.sub_dirname = "control" 
        if self.is_train:
            self.n_simu = 20000
        else:
            self.n_simu = 50
        self.RESCALER = torch.tensor([2, 18, 20, 16, 20, 1]).reshape(1, 6, 1, 1) 

    def __len__(self):
        return self.n_simu

    def __getitem__(self, sim_id):
        if self.is_train:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                             dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                             dtype=torch.float) # 33, 8
            s = s[:, 1]/s.sum(-1) # 33
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) # 33, 64, 64
            state = torch.cat((d, v, c, s), dim=0)[:, :32] # 6, 32, 64, 64
        
            data = (
                state.permute(1, 0, 2, 3) / self.RESCALER, # 32, 6, 64, 64
                sim_id,
            )
        else:
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                            dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                            dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                            dtype=torch.float).permute(2,3,0,1)
            s = torch.tensor(np.load(os.path.join(self.root, self.dirname, self.sub_dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                            dtype=torch.float)
            s = s[:, 1]/s.sum(-1)
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) 
            state = torch.cat((d, v, c, s), dim=0)[:, :256] # 6, 256, 64, 64
            # state = state.permute(1, 0, 2, 3) # shape: 256, 6, 64, 64
            # state = state[::8] # 32, 6, 64, 64
            data = (
                state.permute(1, 0, 2, 3), # 256, 6, 64, 64, not rescaled
                sim_id,
            )
        
        return data
    
if __name__ == "__main__":
    dataset = Jellyfish(
        dataset="jellyfish",
        dataset_path="/weilong/weilong/data/control/jellyfish/",
        steps=20,
        time_interval=1,
        is_train=True,
        is_testdata = True,
    )
    print(len(dataset))
    data = dataset[1999]
    print("state shape: ", data[0].shape)
    print("bd_mask_offset shape: ", data[1].shape)
    print("thetas shape: ", data[2].shape)