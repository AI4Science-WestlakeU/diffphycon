import argparse
import datetime
import numpy as np
import itertools
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
from functools import reduce
import operator
from datetime import datetime
from scripts_SAC.sac_2d import SAC
from scripts_SAC.utils import *
from ddpm.data_2d import Smoke
from ddpm.diffusion_2d import Simulator
from dataset.data_2d_sac import SAC_Smoke


def cycle(dl):
    while True:
        for data in dl:
            yield data
            

def load_data(args, batch_size, is_train):
    dataset = Smoke(
        dataset_path=args.dataset_path,
        is_train=is_train,
    )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 16)
    
    return dataset, data_loader


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print("The folder does not exist. Creating...")
        mkdir(folder_path)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--dataset_path', default="/data/2d/", type=str, 
                    help='path to dataset') 
parser.add_argument('--memory_path', default="/data/2d/rl/", type=str, 
                    help='path to memory') 
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--frames', type=int, default=32, 
                    help='number of time steps in each training sample')
parser.add_argument('--surrogate_model_path', type=str, default='/cp/surrogate_model.pth',
                    help='path to surrogate model')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')

parser.add_argument("--online", action='store_true', help='online?')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Deterministic)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy (default: True)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='G',
                    help='discount factor for reward (default: 1)')
parser.add_argument('--tau', type=float, default=0.05, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')
parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                    help='learning rate of critic loss (default: 0.0003)')
parser.add_argument('--ent_lr', type=float, default=0.0003, metavar='G',
                    help='learning rate of entropy loss (default: 0.0003)')
parser.add_argument('--policy_lr', type=float, default=0.0003, metavar='G',
                    help='learning rate of policy loss (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=1, metavar='G',
                    help='Temperature parameter \alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust \alpha (default: True)')
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                    help='batch size for parameters update (default: 32)')
parser.add_argument('--num_episode', type=int, default=1500, metavar='N',
                    help='maximum number of steps (default: 1500)')
parser.add_argument('--warm_up_episodes', type=int, default=0, metavar='N',
                    help='number of warm up episodes (default: 0)')
parser.add_argument('--hidden_size', type=int, default=2048, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=20, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=5, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=5000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--train_batch_size', type=int, default=10, metavar='N',
                    help='training batch size for online interaction (default: 50)') 
parser.add_argument('--test_batch_size', type=int, default=25, metavar='N',
                    help='test batch size (default: 25)')
parser.add_argument('--test_batch', type=int, default=1, metavar='N',
                    help='number of test batches')
args = parser.parse_args()

print(args)

# Environment
device = torch.device("cuda")

num_t=args.frames - 1
s=args.image_size

# Agent: input normalized data
agent = SAC(4, 2, args)
print('Number of parameters: ', count_params(agent.policy) + count_params(agent.critic) + count_params(agent.critic_target))

# Online data
train_dataset, train_dl = load_data(args, args.train_batch_size, is_train=True)
train_dl = cycle(train_dl)
RESCALER = train_dataset.RESCALER # surrogate model has the same RESCALER
vel_init=torch.tensor(np.load(args.dataset_path+'/intial_vel.npy'), device=device).permute(2,0,1)[:,::2,::2]/RESCALER[0,1:3].to(device)

# Test data
test_dataset, test_dl = load_data(args, args.test_batch_size, is_train=False)
test_dl = cycle(test_dl)

# Surrogate model
simulator = Simulator(
        dim = args.image_size,
        out_dim = 4,
        dim_mults = (1, 2, 4),
        channels = 6
)
simulator.to(device)
simulator.load_state_dict(torch.load(args.surrogate_model_path))
x_test_RESCALER = RESCALER.to(device)
y_test_RESCALER = torch.cat((RESCALER[:,:3], RESCALER[:,[-1]]), dim=1).to(device)

# Memory
if not os.path.exists(os.path.join(args.memory_path, 'online/')):
    os.makedirs(os.path.join(args.memory_path, 'online/'))
clear_folder(os.path.join(args.memory_path, 'online/'))
memory_dataset = SAC_Smoke(RESCALER=RESCALER,
                            num_t=args.frames,
                            offline_dir = args.memory_path, 
                            online_dir = os.path.join(args.memory_path, 'online/'),
                            )
memory_dl = DataLoader(memory_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 16)
memory = cycle(memory_dl)

# Logs
logs=dict()

logs['critic_1_loss']=[]
logs['critic_2_loss']=[]
logs['policy_loss']=[]
logs['ent_loss']=[]
logs['alpha']=[]
logs['avg_reward']=[]
logs_txt = []
logs_txt.append(str(args))
date = datetime.now()
print('Model saved at: /cp/SAC_{}_{}'.format(date, args.seed))

# Training Loop
updates = 0
mask_batch = torch.ones((1, num_t, 1, 1)).expand(-1, -1, s, s)
mask_batch[:, -1] = 0
mask_batch = mask_batch.numpy()
num_online_data = 0
warm_up_episodes = args.warm_up_episodes

for i_episode in tqdm(range(args.num_episode)):

    t1 = datetime.now()

    if args.online:
        if i_episode >= warm_up_episodes:
            state, _, _, _ = next(train_dl) # rescaled
            state_t=state[:,0].to(device) # b, 6, 64, 64
            vel_init_batch=vel_init.unsqueeze(0).repeat(state.shape[0],1,1,1).to(device)
            state_t[:,1:3]=vel_init_batch # b, 6, 64, 64
            time = torch.zeros(state.shape[0], device=device).reshape(-1, 1, 1, 1).expand(-1, -1, s, s)
            action = torch.tensor(agent.select_action(torch.cat((state_t[:,:3],time/args.frames),dim=1).cpu()), device=device) 
            action[:,:,8:56,8:56] = 0
            state_t[:,3:5]=action
            
            now_state_all = []
            action_all = []
            next_state_all = []
            mask_batch_all = []
            for t in range(1, num_t+1):
                now_state_all.append(torch.cat((state_t[:,:3],state_t[:,[-1]], time), dim=1).detach().cpu().numpy())
                action_all.append(action.detach().cpu().numpy())
                state_t = simulator(state_t).data # b, 4, 64, 64
                next_state_all.append(torch.cat((state_t[:,:3],state_t[:,[-1]], time), dim=1).detach().cpu().numpy())
                time = t*torch.ones(state.shape[0], device=device).reshape(-1, 1, 1, 1).expand(-1, -1, s, s) 
                action = torch.tensor(agent.select_action(torch.cat((state_t[:,:3],time/args.frames), dim=1).cpu()), device=device) 
                action[:,:,8:56,8:56] = 0
                state_t = torch.cat((state_t[:,:3], action, state_t[:,[-1]]), dim=1)
                mask_batch_all.append(np.repeat(mask_batch[:, [t-1]], args.train_batch_size, axis=0))
                
            now_state_all = np.stack(now_state_all).reshape(-1, 5, s, s)
            action_all = np.stack(action_all).reshape(-1, 2, s, s)
            next_state_all = np.stack(next_state_all).reshape(-1, 5, s, s)
            mask_batch_all = np.stack(mask_batch_all).flatten()
            
            for k in range(now_state_all.shape[0]):
                # print(num_online_data + k)
                np.savez_compressed(os.path.join(args.memory_path, 'online/{}.npz'.format(num_online_data + k)),
                                    now_state_all = now_state_all[k],
                                    action_all = action_all[k],
                                    next_state_all = next_state_all[k],
                                    mask_batch_all = mask_batch_all[k])
            num_online_data += now_state_all.shape[0]
            # print("num_online_data: ", num_online_data)
            # print(len(os.listdir(os.path.join(args.memory_path, 'online/'))))
            # print(os.listdir(os.path.join(args.memory_path, 'online/')))
        
    t2 = datetime.now()
    # print("Time: {}".format(t2-t1))
    # Number of updates per step in environment
    for i in range(args.updates_per_step):
    # Update parameters of all the networks
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, policy_para = agent.update_parameters(memory, args.batch_size, updates)

    t3 = datetime.now()
    updates += 1    

    if (i_episode + 1) % 10  == 0:
        print("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        logs_txt.append("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, '/cp/SAC_{}_{}_{}'.format(date, args.seed, i_episode))

    if (i_episode + 1) % 40 == 0 and args.eval == True:
        avg_reward = 0.
        test_batch = 0
        state, _, _, _ = next(test_dl) # not rescaled
        state = state[:,::8].to(device) / RESCALER.reshape(1,1,-1,1,1).to(device)
        state_t=state[:,0] # b, 6, 64, 64
        vel_init_batch=vel_init.unsqueeze(0).repeat(state.shape[0],1,1,1).to(device)
        state_t[:,1:3]=vel_init_batch # b, 6, 64, 64
        time = torch.zeros(state.shape[0], device=device).reshape(-1, 1, 1, 1).expand(-1, -1, s, s)
        action = torch.tensor(agent.select_action(torch.cat((state_t[:,:3],time/args.frames),dim=1).cpu()), device=device).detach()
        action[:,:,8:56,8:56] = 0
        state_t[:,3:5]=action
        
        for t in range(1, num_t+1):
            state_t = simulator(state_t).data # b, 4, 64, 64
            time = t*torch.ones(state.shape[0], device=device).reshape(-1, 1, 1, 1).expand(-1, -1, s, s)
            action = torch.tensor(agent.select_action(torch.cat((state_t[:,:3],time/args.frames),dim=1).cpu()), device=device).detach()
            action[:,:,8:56,8:56] = 0
            state_t = torch.cat((state_t[:,:3], action, state_t[:,[-1]]), dim=1)

        print(action.abs().mean())
        avg_reward += state_t[:, -1].mean().item()
        avg_reward /= args.test_batch

        print("Objective_function: {:1.6f}".format(avg_reward))
        logs_txt.append("Objective_function: {:1.6f}".format(avg_reward))

        logs['critic_1_loss'].append(critic_1_loss)
        logs['critic_2_loss'].append(critic_2_loss)
        logs['policy_loss'].append(policy_loss)
        logs['ent_loss'].append(ent_loss)
        logs['alpha'].append(alpha)
        logs['avg_reward'].append(-avg_reward)
        
        torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, '/cp/SAC_{}_{}_{}'.format(date, args.seed, i_episode))
        with open('/cp/SAC_{}_{}.txt'.format(date, args.seed), 'w') as f:
            for item in logs_txt:
                f.write("%s\n" % item)

torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, '/cp/SAC_{}_{}_{}'.format(date, args.seed, i_episode))
