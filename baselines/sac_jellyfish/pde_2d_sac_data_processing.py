import argparse
import datetime
import numpy as np
import itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import random
import h5py
from datetime import datetime
from scripts.sac_2d import SAC
from scripts.utils import *
from scripts.replay_memory import ReplayMemory
from sim_ppl_2d import *


def cycle(dl):
    while True:
        for data in dl:
            yield data
            

def load_data(args, batch_size, is_train):
    dataset = Jellyfish(
        dataset="jellyfish", 
        dataset_path=args.dataset_path,
        time_steps=args.time_steps, 
        steps=args.frames, 
        time_interval=1, 
        is_train=is_train, 
        show_missing_files=False, 
        is_traj=False, 
        is_testdata=args.is_testdata,
        for_pipeline=True
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 16)
    
    return train_loader

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--dataset_path', default="data/jellyfish", type=str, help='path to dataset') 
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--time_steps', type=int, default=40, help='number of all time steps in each simulation')
parser.add_argument('--frames', type=int, default=20, help='number of time steps in each training sample')
parser.add_argument('--m', type=int, default=20, help='number of points on each ellipse wing of jellyfish')
parser.add_argument('--memory_batch_size', type=int, default=2800, metavar='N',
                    help='batch size for memory pushing (default: 2800)')
args = parser.parse_args()

print(args)

device = torch.device("cuda")

num_t=args.frames - 1
s=args.image_size

mask_batch = np.ones((1, num_t))
mask_batch[:, -1] = 0

ppl = build_ppl_new(args)
dataloader = load_data(args, args.memory_batch_size, is_train=True)
print("number of batch to push to memory: ", len(dataloader))
start = datetime.now()
for i, data in tqdm(enumerate(dataloader)):
    print("i: ", i)
    state, mask_offsets, thetas, bd_mask_offset_0, sim_id, time_id = data
    thetas = thetas.to(device)
    state_pad = torch.zeros(state.shape[0], args.frames, 3, s, s).to(device)
    mask_offsets_pad = torch.zeros(mask_offsets.shape[0], args.frames, 3, s, s).to(device)
    bd_mask_offset_0_pad = torch.zeros(bd_mask_offset_0.shape[0], 3, s, s).to(device)
    state_pad = state.to(device)
    mask_offsets_pad[:,:,:,1:-1,1:-1] = mask_offsets.to(device)
    bd_mask_offset_0_pad[:,:,1:-1,1:-1] = bd_mask_offset_0.to(device)

    force = []
    cond_T = []
    time = []
    for t in range(1, num_t+1):
        state_t = state_pad[:,t,:,:,:]
        mask_offsets_t = mask_offsets_pad[:,t,:,:,:]
        force_t = ppl.run(state_t, mask_offsets_t) # output force of the current time step
        if t != num_t:
            force.append((num_t + 1 - t) * force_t.cpu().numpy())
            cond_T.append(0 * force_t.cpu().numpy())
        else:
            force.append(force_t.detach().cpu().numpy())
            cond_T.append(((mask_offsets_pad[:,-1] - mask_offsets_pad[:,0])**2).sum((1,2,3)).detach().cpu().numpy())
        time.append((t - 1) * np.ones_like(force[-1]))
    now_state = torch.cat((state_pad[:,:num_t], mask_offsets_pad[:,:num_t]), dim=2).cpu().numpy().reshape(-1,6,s,s)
    next_state = torch.cat((state_pad[:,1:], mask_offsets_pad[:,1:]), dim=2).cpu().numpy().reshape(-1,6,s,s)
    now_state = np.concatenate((now_state, np.repeat(mask_offsets_pad[:,[0]].cpu().numpy(), num_t, 1).reshape(-1,3,s,s)), 1)
    next_state = np.concatenate((next_state, np.repeat(mask_offsets_pad[:,[0]].cpu().numpy(), num_t, 1).reshape(-1,3,s,s)), 1)
    
    now_state_all = now_state
    thetas_all = thetas[:, 1:].cpu().numpy().flatten()
    cond_theta_all = thetas[:, [0]].repeat(1, num_t).cpu().numpy().flatten()
    thetas_delta_all = (thetas[:, 1:] - thetas[:, :-1]).cpu().numpy().flatten()
    force_all = np.array(force).transpose((1,0)).flatten()
    cond_T_all = np.array(cond_T).transpose((1,0)).flatten()
    time_all = np.array(time).transpose((1,0)).flatten()
    next_state_all = next_state
    mask_batch_all = np.repeat(mask_batch, args.memory_batch_size, 0).flatten()
    # print("now_state_all.shape: ", now_state_all.shape)
    # print("thetas_all.shape: ", thetas_all.shape)
    # print("force_all.shape: ", force_all.shape)
    # print("cond_T_all.shape: ", cond_T_all.shape)
    # print("next_state_all.shape: ", next_state_all.shape)
    # print("mask_batch_all.shape: ", mask_batch_all.shape)
    for k in range(int(now_state_all.shape[0])):
        # print(i*args.memory_batch_size*num_t + k)
        np.savez_compressed("data/memory_delta_t_20/{}.npz".format(i*args.memory_batch_size*num_t + k), cond_theta_all=cond_theta_all[k],\
                            time_all=time_all[k], now_state_all=now_state_all[k], thetas_all=thetas_all[k], thetas_delta_all=thetas_delta_all[k], force_all=force_all[k],\
                            cond_T_all=cond_T_all[k], next_state_all=next_state_all[k], mask_batch_all=mask_batch_all[k])
    end = datetime.now()
    print('time: ', end-start)
    start = datetime.now()
