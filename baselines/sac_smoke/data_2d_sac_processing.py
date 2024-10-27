import argparse
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import random
import h5py
from datetime import datetime
from scripts.sac_2d import SAC
from scripts.utils import *
from dataset.data_surrogate_model_2d import SimulatorData


def cycle(dl):
    while True:
        for data in dl:
            yield data
            

def load_data(args, batch_size, is_train):
    dataset = SimulatorData(
        dataset_path=args.dataset_path,
        size = args.image_size,
        is_train= True,
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 16)
    
    return train_loader

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--dataset_path', default="/data/2d/", type=str, help='path to dataset') 
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
parser.add_argument('--memory_batch_size', type=int, default=3000, metavar='N',
                    help='batch size for memory pushing (default: 3000)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')

args = parser.parse_args()

print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda")

num_t = 32
s = args.image_size

dataloader = load_data(args, args.memory_batch_size, is_train=True)
print("number of batch to push to memory: ", len(dataloader))
start = datetime.now()
for i, data in tqdm(enumerate(dataloader)):
    print("i: ", i)
    x, y, time = data
    x, y, time = x.numpy(), y.numpy(), time.reshape(-1, 1, 1, 1).expand(-1, -1, s, s).numpy()  

    now_state = np.concatenate((np.concatenate((x[:,:3], x[:,[-1]]), axis=1), time), 1) # d, v1, v2, s, t
    action = x[:, 3:5] # c1, c2
    next_state = np.concatenate((y, time+1), 1)
    mask_batch = np.ones_like(time)
    mask_batch[time == num_t - 2] = 0

    # print("now_state.shape: ", now_state.shape)
    # print("action.shape: ", action.shape)
    # print("next_state.shape: ", next_state.shape)
    # print("mask_batch.shape: ", mask_batch.shape)

    for k in range(int(now_state.shape[0])):
        # print(i*args.memory_batch_size + k)
        np.savez_compressed("/data/2d/rl/memory_{}.npz".format(i*args.memory_batch_size + k), 
                            now_state_all = now_state[k], 
                            action_all = action[k], 
                            next_state_all = next_state[k], 
                            mask_batch_all = mask_batch[k])
    end = datetime.now()
    print('time: ', end-start)
    start = datetime.now()