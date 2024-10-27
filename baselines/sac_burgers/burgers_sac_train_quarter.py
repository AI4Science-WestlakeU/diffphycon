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
from scripts.sac import SAC
from scripts.utils import *
from scripts.replay_memory import ReplayMemory
from generate_burgers import burgers_numeric_solve
from model.pde_1d_surrogate_model.burgers_operator import Simu_surrogate_model


def cycle(dl):
    while True:
        for data in dl:
            yield data
            

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument("--online", action='store_true', help='online?')
parser.add_argument("--surrogate", action='store_true', help='use surrogate model?')
parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
parser.add_argument('--reward_f', type=float, default=0, metavar='G',
                    help='weight of energy')
parser.add_argument('--grid_size', default=128, type=int, 
                    help = 'grid size (default: 128)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Deterministic)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 20 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='G',
                    help='discount factor for reward (default: 1)')
parser.add_argument('--tau', type=float, default=0.05, metavar='G',
                    help='target smoothing coefficient(\tau) (default: 0.005)')
parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                    help='learning rate of critic loss (default: 0.0003)')
parser.add_argument('--ent_lr', type=float, default=0.003, metavar='G',
                    help='learning rate of entropy loss (default: 0.0003)')
parser.add_argument('--policy_lr', type=float, default=0.003, metavar='G',
                    help='learning rate of policy loss (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.02, metavar='G',
                    help='Temperature parameter \alpha determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust \alpha (default: True)')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=8192, metavar='N',
                    help='batch size for parameters update (default: 8192)')
parser.add_argument('--num_episode', type=int, default=1500, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=4096, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=15, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--train_batch_size', type=int, default=1, metavar='N',
                    help='training batch size (default: 5)')
parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                    help='test batch size (default: 256)')
parser.add_argument('--test_batch', type=int, default=1, metavar='N',
                    help='number of test batches')
args = parser.parse_args()

print(args)

# Environment
device = torch.device('cuda', args.gpu)
reward_f = args.reward_f

s = args.grid_size
xmin = 0.0; xmax = 1.0
delta_x = (xmax-xmin)/(s+1)
X = np.linspace(xmin+delta_x,xmax-delta_x,s)

num_t=10
tmin = 0.0; tmax = 1
delta_t = (tmax-tmin)/num_t
t = torch.linspace(tmin,tmax,num_t+1)
T = np.linspace(tmin,tmax,num_t+1)

# Agent
agent = SAC(s+s+1, int(s/2), args)

pde_1d_surrogate_model_checkpoint="/checkpoints/pde_1d/full_ob_partial_ctr_1-step"
milestone=500
simu_surrogate_model=Simu_surrogate_model(path=pde_1d_surrogate_model_checkpoint,device=device,s_ob=128,milestone=milestone)
mse=torch.nn.MSELoss()

# Memory
memory = ReplayMemory(args.replay_size)
mask_batch = np.ones((num_t))
mask_batch[-1] = 0

dataset = h5py.File("/data/data_1d_burgers/free_u_f_1e5_front_rear_quarter/burgers_train.h5", 'r')['train']
u_data, f_data = torch.tensor(np.array(dataset['pde_11-128']), device=device).float(), torch.tensor(np.array(dataset['pde_11-128_f']), device=device).float()
Ndata = u_data.shape[0]
u_target = u_data[:, -1]
# u_data[:, -1] = u_data[:, -1] + torch.randn(u_data[:, -1].shape, device=device) * 0.01
for i in range(num_t):
    state = torch.cat((i*torch.ones_like(u_data[:, 0, [0]]), u_data[:, i], u_target), -1)  
    action = f_data[:, i]  
    action = torch.cat((action[:, :int(s/4)], action[:, int(3*s/4):]), -1)
    if True:  
        uT_reward = (-1) * torch.sum((u_target-u_data[:, i+1])**2, -1) / s
    else:  
        uT_reward = 0  
    f_reward = torch.sum(f_data[:, i]**2, -1) 
    reward = uT_reward + (-reward_f) * f_reward  
    next_state = torch.cat(((i+1)*torch.ones_like(u_data[:, 0, [0]]), u_data[:, i+1], u_target), -1)  
    memory.push_batch(list(zip(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(),\
                    next_state.cpu().numpy(), np.repeat(mask_batch[i], Ndata))))

# Train data
train_dl = DataLoader(TensorDataset(torch.cat((u_data.cpu(), f_data.cpu()), 1)), batch_size = args.train_batch_size, shuffle = True)
train_dl = cycle(train_dl)

# Test data
test_dataset = h5py.File("/data/data_1d_burgers/free_u_f_1e5_front_rear_quarter/burgers_test.h5", 'r')['test']
test_u_data, test_f_data = torch.tensor(np.array(test_dataset['pde_11-128'])).float(), torch.tensor(np.array(test_dataset['pde_11-128_f'])).float()
test_u_data, test_f_data = test_u_data.cpu(), test_f_data.cpu()
test_dl = DataLoader(TensorDataset(torch.cat((test_u_data, test_f_data), 1)), batch_size = args.test_batch_size, shuffle = True)

# Logs
logs=dict()
logs['critic_1_loss']=[]
logs['critic_2_loss']=[]
logs['policy_loss']=[]
logs['ent_loss']=[]
logs['alpha']=[]
logs['avg_reward']=[]
logs['avg_uT_reward']=[]
logs['avg_f_reward']=[]
logs['avg_reward_surrogate']=[]
logs['avg_uT_reward_surrogate']=[]
logs['avg_f_reward_surrogate']=[]
logs['state_loss']=[]
logs['state_relative_loss']=[]
logs_txt = []
date = datetime.now()

# Training Loop
updates = 0

for i_episode in tqdm(range(args.num_episode)):
    episode_reward = 0
    episode_steps = 0
    t1 = datetime.now()

    if args.online:
        data = next(train_dl)[0]
        u_data = np.array(data[:, :num_t])
        state = np.concatenate((np.zeros_like(u_data[:,0,[0]]), u_data[:,0], u_data[:,-1]), -1).reshape(-1,s+s+1)
        batch_size = data.shape[0]
        action = np.zeros((batch_size, 1, s))
        for i in range(num_t):
            action_quarter = agent.select_action(state)
            action[:, 0, :int(s/4)] = action_quarter[:, :int(s/4)]
            action[:, 0, int(3*s/4):] = action_quarter[:, int(s/4):]
            if args.surrogate: 
                next_state = simu_surrogate_model.simulation(ut=torch.tensor(state[:,None,1:s+1], device=device).float(), ft=torch.tensor(action, device=device).float()) \
                                        .cpu().numpy().reshape(batch_size, s)
            else:
                next_state = burgers_numeric_solve(torch.tensor(state[:,1:s+1], device=device), torch.tensor(action, device=device).float(), visc=0.01, T=delta_t, dt=1e-4, num_t=1)\
                                [torch.arange(batch_size), torch.arange(batch_size), -1].cpu().numpy().reshape(batch_size, s)
            
            if True:
                uT_reward = (-1) * np.sum((next_state-u_data[:,-1])**2, -1) / s
            else:
                uT_reward = 0
            f_reward = np.sum(action**2, -1).reshape(-1) 
            reward = uT_reward + (-reward_f) * f_reward
            next_state = np.concatenate(((i+1)*np.ones_like(next_state[:,[0]]), next_state, u_data[:,-1]), -1)
            state = next_state

            mask = 0 if i == num_t-1 else 1
            memory.push_batch(list(zip(state.reshape(batch_size, -1), action_quarter.reshape(batch_size, -1), reward.reshape(batch_size),\
                        next_state.reshape(batch_size, -1), np.repeat(mask, batch_size))))

            state = next_state

    t2 = datetime.now()
    for i in range(args.updates_per_step):
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, policy_para = agent.update_parameters(memory, args.batch_size, updates)

    t3 = datetime.now()
    updates += 1    

    if (i_episode + 1) % 5 == 0:
        print("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        logs_txt.append("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        
    if (i_episode + 1) % 25 == 0 and args.eval == True:
        avg_reward = 0.
        avg_uT_reward = 0.
        avg_f_reward = 0.
        avg_reward_surrogate = 0.
        avg_uT_reward_surrogate = 0.
        avg_f_reward_surrogate = 0.
        test_batch = 0
        for data in test_dl:
            break
        data = data[0]
        u_data = np.array(data[:, :num_t])
        state = np.concatenate((np.zeros_like(u_data[:,0,[0]]), u_data[:,0], u_data[:,-1]), -1).reshape(-1,s+s+1)
        state_surrogate = state
        state_true = state
        batch_size = data.shape[0]
        action = np.zeros((batch_size, 1, s))
        action_surrogate = np.zeros((batch_size, 1, s))
        next_state_surrogat_all = []
        next_state_true_all = []

        for i in range(num_t):
            # solver:
            action_quarter = agent.select_action(state, eval=True)
            action[:, 0, :int(s/4)] = action_quarter[:, :int(s/4)]
            action[:, 0, int(3*s/4):] = action_quarter[:, int(s/4):]
            next_state = burgers_numeric_solve(torch.tensor(state[:,1:s+1], device=device), torch.tensor(action, device=device).float(), visc=0.01, T=delta_t, dt=1e-4, num_t=1)\
                            [torch.arange(batch_size), torch.arange(batch_size), -1].cpu().numpy().reshape(batch_size, s)     
            if i == num_t-1:
                uT_reward = (-1) * np.sum((next_state-u_data[:,-1])**2) / batch_size / s
            else:
                uT_reward = 0
            f_reward = np.sum(action**2) / batch_size 
            reward = uT_reward + (-reward_f) * f_reward
            next_state = np.concatenate(((i+1)*np.ones_like(next_state[:,[0]]), next_state, u_data[:,-1]), -1)
            state = next_state

            avg_reward += reward
            avg_uT_reward += uT_reward
            avg_f_reward += f_reward

            # surrogate model:
            action_quarter = agent.select_action(state_surrogate, eval=True)
            action_surrogate[:, 0, :int(s/4)] = action_quarter[:, :int(s/4)]
            action_surrogate[:, 0, int(3*s/4):] = action_quarter[:, int(s/4):]
            next_state_surrogate = simu_surrogate_model.simulation(ut=torch.tensor(state_surrogate[:,None,1:s+1], device=device), ft=torch.tensor(action_surrogate, device=device).float()) \
                                    .cpu().numpy().reshape(batch_size, s)
            #ut [batch_size,1,ns], ft [batch_size,1,ns=128], u_{t+1} [batch_size,1,ns]
            next_state_true = burgers_numeric_solve(torch.tensor(state_true[:,1:s+1], device=device), torch.tensor(action_surrogate, device=device).float(), visc=0.01, T=delta_t, dt=1e-4, num_t=1)\
                            [torch.arange(batch_size), torch.arange(batch_size), -1].cpu().numpy().reshape(batch_size, s)   
            next_state_surrogat_all.append(next_state_surrogate)
            next_state_true_all.append(next_state_true)
            if i == num_t-1:
                uT_reward = (-1) * np.sum((next_state_true-u_data[:,-1])**2) / batch_size / s
            else:
                uT_reward = 0
            f_reward = np.sum(action_surrogate**2) / batch_size 
            reward = uT_reward + (-reward_f) * f_reward
            next_state_surrogate = np.concatenate(((i+1)*np.ones_like(next_state[:,[0]]), next_state_surrogate, u_data[:,-1]), -1)
            next_state_true = np.concatenate(((i+1)*np.ones_like(next_state[:,[0]]), next_state_true, u_data[:,-1]), -1)
            state_surrogate = next_state_surrogate
            state_true = next_state_true

            avg_reward_surrogate += reward
            avg_uT_reward_surrogate += uT_reward
            avg_f_reward_surrogate += f_reward

        state_loss=mse(torch.tensor(np.array(next_state_surrogat_all)), torch.tensor(np.array(next_state_true_all))) 
        state_relative_loss = torch.norm(torch.tensor(np.array(next_state_surrogat_all)) - torch.tensor(np.array(next_state_true_all))) / torch.norm(torch.tensor(np.array(next_state_true_all))) 
        avg_reward /= args.test_batch
        avg_uT_reward /= args.test_batch
        avg_f_reward /= args.test_batch
        avg_reward_surrogate /= args.test_batch
        avg_uT_reward_surrogate /= args.test_batch
        avg_f_reward_surrogate /= args.test_batch
        
        print("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f} | Objective_function: {:1.6f}={:1.6f}+{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, -avg_reward, -avg_uT_reward, avg_f_reward))
        logs_txt.append("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f} | Objective_function: {:1.6f}={:1.6f}+{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, -avg_reward, -avg_uT_reward, avg_f_reward))
        print("state loss:{:1.6f}, relative state loss:{:1.6f}, Objective_function_surrogate: {:1.6f}={:1.6f}+{:1.6f}".format(state_loss, state_relative_loss, -avg_reward_surrogate, -avg_uT_reward_surrogate, avg_f_reward_surrogate))
        logs_txt.append("state loss:{:1.6f}, relative state loss:{:1.6f}, Objective_function_surrogate: {:1.6f}={:1.6f}+{:1.6f}".format(state_loss, state_relative_loss, -avg_reward_surrogate, -avg_uT_reward_surrogate, avg_f_reward_surrogate))
        
        logs['critic_1_loss'].append(critic_1_loss)
        logs['critic_2_loss'].append(critic_2_loss)
        logs['policy_loss'].append(policy_loss)
        logs['ent_loss'].append(ent_loss)
        logs['alpha'].append(alpha)
        logs['avg_reward'].append(-avg_reward)
        logs['avg_uT_reward'].append(-avg_uT_reward)
        logs['avg_f_reward'].append(-avg_f_reward)
        logs['avg_reward_surrogate'].append(-avg_reward_surrogate)
        logs['avg_uT_reward_surrogate'].append(-avg_uT_reward_surrogate)
        logs['avg_f_reward_surrogate'].append(-avg_f_reward_surrogate)
        logs['state_loss'].append(state_loss)
        logs['state_relative_loss'].append(state_relative_loss)
        
        torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, 'cp/SAC_policy_track_quarter{}_{}_{}'.format(date, args.seed, i_episode))
        with open('cp/SAC_policy_track_quarter{}_{}.txt'.format(date, args.seed), 'w') as f:
            for item in logs_txt:
                f.write("%s\n" % item)
        print('Checkpoint: cp/SAC_policy_track_quarter{}_{}_{}'.format(date, args.seed, i_episode))

torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, 'cp/SAC_policy_track_quarter{}_{}_{}'.format(date, args.seed, i_episode))