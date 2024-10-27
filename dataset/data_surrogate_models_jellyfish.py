# %%
import numpy as np
import torch
# from torch_geometric.data import Dataset, Data
from torch.utils.data import Dataset
import pdb
import os
import time
import pickle
        

# TODO: replace Dataset of torch_geometric.data by Dataset of from torch.utils.data
# dataset of surrogate force model
class ForceData(Dataset):
    def __init__(
        self,
        dataset_path,
        n_simu,
        train_split_ratio,
        transform=None,
        pre_transform=None,
        is_train=True
    ):
        self.root = dataset_path
        self.n_simu = n_simu
        self.train_split_ratio = train_split_ratio
        self.n_simu_train = int(self.n_simu * self.train_split_ratio)
        self.n_simu_test = self.n_simu - self.n_simu_train
        self.is_train = is_train
        self.time_stamps = 40
        self.critic_val = 50
        # normalization_filename = os.path.join(self.root, "normalization_max_min.pkl")
        # if os.path.isfile(normalization_filename): 
        #     normdict = pickle.load(open(normalization_filename, "rb"))
        #     self.vx_max = normdict["vx_max"]
        #     self.vx_min = normdict["vx_min"]
        #     self.vy_max = normdict["vy_max"]
        #     self.vy_min = normdict["vy_min"]
        #     self.p_max = normdict["p_max"]
        #     self.p_min = normdict["p_min"] 

        super(ForceData, self).__init__(self.root, transform, pre_transform)
    
    def len(self):
        if self.is_train: 
            return self.time_stamps * self.n_simu_train
        else:
            return self.time_stamps * self.n_simu_test

    def locate_index(self, idx):
        if self.is_train: 
            sim_id, time_id = divmod(idx, self.time_stamps)
            assert 0<= sim_id < self.n_simu_train
        else:
            sim_id, time_id = divmod(idx, self.time_stamps)
            sim_id += self.n_simu_train
            assert self.n_simu_train <= sim_id < self.n_simu 
        return sim_id, time_id
    
    
    def get(self, idx):        
        sim_id, time_id = self.locate_index(idx)
        force = np.load(os.path.join(self.root, "forces", "sim_{:06d}.npy").format(sim_id))[time_id]
        force = np.sum(force, axis = 0)[:2] # summation over num of boundaries and then only takes x(drag) and y(lift) force
        pressure = torch.FloatTensor(np.load(os.path.join(self.root, "states", "sim_{:06d}.npz".format(sim_id)))["a"][time_id][2,:,:]).unsqueeze(-1) # [64, 64, 1]
        # pressure = (torch.clamp((pressure - self.p_min) / (self.p_max - self.p_min), 0, 1) - 0.5) * 2
        # make elements whose value is greater than 2000 or smaller than -2000 to be the average value of the rest
        
        pressure[pressure > self.critic_val] = torch.mean(pressure[torch.abs(pressure) <= self.critic_val])
        pressure[pressure < -self.critic_val] = torch.mean(pressure[torch.abs(pressure) <= self.critic_val])

        mask_offset = torch.FloatTensor(np.load(os.path.join(self.root, 'bdry_merged_mask_offsets', 'sim_{:06d}.npz'.format(sim_id)))["a"][time_id]) 
        image_size = mask_offset.shape[0]
        mask_offset_pad = torch.zeros((image_size + 2, image_size + 2, 3))
        mask_offset_pad[1:-1, 1:-1, :] = mask_offset
        x = np.concatenate((pressure, mask_offset_pad), axis=2)
        x = torch.cat((pressure, mask_offset_pad), 2)
        x = x.float().permute(2, 0, 1)
        force = torch.from_numpy(force).float()
        x[torch.isnan(x)] = 0 
        force[torch.isnan(force)] = 0 # 
        data = (
            x, # [4, 64, 64], 4 means pressure ([1, 64, 64]) and mask_offset[3, 64, 64]
            force[0].unsqueeze(0) # [1] force in x direction
        )  
            
        return data

# dataset of surrogate simulator model
class SimulatorData(Dataset):
    def __init__(
        self,
        dataset_path,
        n_simu,
        train_split_ratio,
        transform=None,
        pre_transform=None,
        is_train=True,
        pred_mask_offset = False,
        only_vis_pressure = False
    ):
        self.root = dataset_path
        self.n_simu = n_simu
        self.train_split_ratio = train_split_ratio
        self.n_simu_train = int(self.n_simu * self.train_split_ratio)
        self.n_simu_test = self.n_simu - self.n_simu_train
        self.is_train = is_train
        self.time_stamps = 40 - 1
        self.critic_val = 50
        self.pred_mask_offset = pred_mask_offset
        self.only_vis_pressure = only_vis_pressure
        if self.only_vis_pressure:
            self.pred_mask_offset = False
        
        # normalization_filename = os.path.join(self.root, "normalization_max_min.pkl")
        # if os.path.isfile(normalization_filename): 
        #     normdict = pickle.load(open(normalization_filename, "rb"))
        #     self.vx_max = normdict["vx_max"]
        #     self.vx_min = normdict["vx_min"]
        #     self.vy_max = normdict["vy_max"]
        #     self.vy_min = normdict["vy_min"]
        #     self.p_max = normdict["p_max"]
        #     self.p_min = normdict["p_min"] 
        super(SimulatorData, self).__init__(self.root, transform, pre_transform)
    
    def len(self):
        if self.is_train: 
            return self.time_stamps * self.n_simu_train
        else:
            return self.time_stamps * self.n_simu_test

    def locate_index(self, idx):
        if self.is_train: 
            sim_id, time_id = divmod(idx, self.time_stamps)
            assert 0<= sim_id < self.n_simu_train
        else:
            sim_id, time_id = divmod(idx, self.time_stamps)
            sim_id += self.n_simu_train
            assert self.n_simu_train <= sim_id < self.n_simu 
        return sim_id, time_id
    
    def get(self, idx):        
        sim_id, time_id = self.locate_index(idx)  
        states_all = torch.FloatTensor(np.load(os.path.join(self.root, "states", "sim_{:06d}.npz".format(sim_id)))["a"]) #3 channels: v_x, v_y, pressure
        if self.only_vis_pressure:
            states_all = states_all[:, 2, :, :].unsqueeze(1)
        states = states_all[time_id]
        mask_offset_all = np.load(os.path.join(self.root, 'bdry_merged_mask_offsets', 'sim_{:06d}.npz'.format(sim_id)))["a"]
        mask_offset = mask_offset_all[time_id] 
        theta_all = torch.from_numpy(
            np.load(os.path.join(self.root, 'bdry_head_thetas/sim_{:06d}.npz'.format(sim_id)))["thetas"]
        )
        theta_delta = (theta_all[time_id + 1] -  theta_all[time_id]).unsqueeze(0).float()
        mask_offset_pad = np.zeros((64, 64, 3))
        mask_offset_pad[1:-1, 1:-1, :] = mask_offset
        mask_offset = torch.FloatTensor(np.transpose(mask_offset_pad, (2, 0, 1)))
        x = torch.cat((states, mask_offset), 0).float()
        y = states_all[time_id + 1].float()
        if self.pred_mask_offset:
            mask_offset_next = mask_offset_all[time_id + 1]
            mask_offset_pad_next = np.zeros((64, 64, 3))
            mask_offset_pad_next[1:-1, 1:-1, :] = mask_offset_next
            mask_offset_next = torch.FloatTensor(np.transpose(mask_offset_pad_next, (2, 0, 1)))
            y = torch.cat((y, mask_offset_next), 0).float()

        x[torch.isnan(x)] = 0 
        y[torch.isnan(y)] = 0 
        data = (
            x, # [6, 64, 64]: 6 means states ([3, 64, 64]) and mask_offset[3, 64, 64], both of current time step
            theta_delta, #[1] angel difference in arc of from time t to t+1
            y # [3, 64, 64], velocity and pressure of the next time step if self.pred_mask_offset==False, otherwise, also include mask_offset
        )  
            
        return data

# dataset of surrogate boundary updater model
class BoundaryUpdaterData(Dataset):
    def __init__(
        self,
        dataset_path,
        n_simu,
        train_split_ratio,
        incremental = True,
        is_train=True,
        transform=None,
        pre_transform=None
    ):
        self.root = dataset_path
        self.n_simu = n_simu
        self.train_split_ratio = train_split_ratio
        self.n_simu_train = int(self.n_simu * self.train_split_ratio)
        self.n_simu_test = self.n_simu - self.n_simu_train
        self.is_train = is_train
        self.time_stamps = 40 - 1
        self.incremental = incremental # whether update boundary mask and offset incrementally
        super(BoundaryUpdaterData, self).__init__(self.root, transform, pre_transform)
    
    def len(self):
        if self.is_train: 
            return self.time_stamps * self.n_simu_train
        else:
            return self.time_stamps * self.n_simu_test

    def locate_index(self, idx):
        if self.is_train: 
            sim_id, time_id = divmod(idx, self.time_stamps)
            assert 0<= sim_id < self.n_simu_train
        else:
            sim_id, time_id = divmod(idx, self.time_stamps)
            sim_id += self.n_simu_train
            assert self.n_simu_train <= sim_id < self.n_simu 
        return sim_id, time_id
    
    def get(self, idx):        
        sim_id, time_id = self.locate_index(idx)  
        mask_offset_all = np.load(os.path.join(self.root, 'bdry_merged_mask_offsets', 'sim_{:06d}.npz'.format(sim_id)))["a"]
        thetas = torch.from_numpy(
            np.load(os.path.join(self.root, 'bdry_head_thetas/sim_{:06d}.npz'.format(sim_id)))["thetas"]
        )
        if self.incremental:
            theta = thetas[time_id].unsqueeze(0).float()
            x = torch.from_numpy(np.transpose(mask_offset_all[time_id], (2, 0, 1))).float() 
        else:
            theta = thetas[0].unsqueeze(0).float()
            x = torch.from_numpy(np.transpose(mask_offset_all[0], (2, 0, 1))).float() 
        theta_next = thetas[time_id+1].unsqueeze(0).float()
        theta_delta = theta_next - theta    
        y = torch.from_numpy(np.transpose(mask_offset_all[time_id+1], (2, 0, 1))).float() 
        x[torch.isnan(x)] = 0 
        y[torch.isnan(y)] = 0 
        data = (
            x, # [3, 62, 62]: boundary mask and offset of time t (or time 0 if self.incremetental==True)
            theta_delta, #[1] angel difference in arc of from time t (or time 0) to t+1
            y # [3, 62, 62], boundary mask and offset of time t+1
        )  
            
        return data
    
if __name__ == "__main__":
    # dataset = ForceData(
    #     is_train= True,
    #     is_testdata = False,
    # )
    dataset = ForceData(
        is_train= False,
        is_testdata = False,
    )
    print(dataset[285].x.shape)
    print(dataset[285].y.shape)