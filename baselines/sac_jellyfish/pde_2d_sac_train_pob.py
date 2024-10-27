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
from scripts.sac_2d_pob import SAC_pob
from scripts.utils import *
from scripts.replay_memory import ReplayMemory
from sim_ppl_2d import *
from data_2d_sac import *


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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 16)
    
    return dataset, train_loader

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

def reg_theta(theta):
    """
    input: theta of shape [batch_size, T]
    output: reg of shape [batch_size]
    compute R(theta) = \sum_{t=0}^{T-2}(theta_{t+1} - theta_t) ** 2
    """
    theta_t = theta[:, :-1] # theta_{t}: [batch_size, T-1]
    theta_t_1 = theta[:, 1:] # theta_{t+1}: [batch_size, T-1]
    reg = torch.sum((theta_t_1 - theta_t) * (theta_t_1 - theta_t), dim=1) # [batch_size]
    return reg


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--dataset_path', default="/data/jellyfish", type=str, help='path to dataset') 
parser.add_argument('--memory_path', default="data/memory_delta_t_20/", type=str, help='path to memory') 
parser.add_argument('--only_vis_pressure', action='store_true', help="whether only observe pressure, only used in simulator model")
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--time_steps', type=int, default=40, help='number of all time steps in each simulation')
parser.add_argument('--frames', type=int, default=20, help='number of time steps in each training sample')
parser.add_argument('--m', type=int, default=20, help='number of points on each ellipse wing of jellyfish')
parser.add_argument('--force_model_checkpoint', type=str, default='checkpoints/force_model/force_model.pth')
parser.add_argument('--simulator_model_checkpoint', type=str, default='checkpoints/simulator_model/simulator_model.pth')
parser.add_argument('--boundary_updater_model_checkpoint', type=str, default='/checkpoints/surrogate_models/boundary_updater_model/2023-12-02_02-39-51/epoch_1_iter_10000.pth')
parser.add_argument('--is_testdata', default=False, type=bool,
                help='whether run mini example data, if True, yes; otherwise, run full data')

parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
parser.add_argument('--memory_batch_size', type=int, default=2800, metavar='N',
                    help='batch size for memory pushing (default: 2800)')
parser.add_argument("--online", action='store_true', help='online?')
parser.add_argument('--reward_alpha', type=float, default=0.002, metavar='G',
                    help='weight of limitation of theta_T in reward')
parser.add_argument('--reg_lambda', type=float, default=-1000, metavar='G',
                    help='weight of regularizer of theta')
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
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                    help='batch size for parameters update (default: 32)')
parser.add_argument('--num_episode', type=int, default=400, metavar='N',
                    help='maximum number of steps (default: 5000)')
parser.add_argument('--warm_up_episodes', type=int, default=0, metavar='N',
                    help='number of warm up episodes (default: 0)')
parser.add_argument('--hidden_size', type=int, default=2048, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=20, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--target_update_interval', type=int, default=15, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--train_batch_size', type=int, default=5, metavar='N',
                    help='training batch size (default: 5)')
parser.add_argument('--test_batch_size', type=int, default=20, metavar='N',
                    help='test batch size (default: 256)')
parser.add_argument('--test_batch', type=int, default=1, metavar='N',
                    help='number of test batches')
args = parser.parse_args()

print(args)

# Environment
device = torch.device("cuda")

num_t=args.frames - 1
s=args.image_size

# Agent: input normalized data
agent = SAC_pob(s, 1, args)

# Online data
train_dataset, train_dl = load_data(args, args.train_batch_size, is_train=True)
train_dl = cycle(train_dl)

# Test data
test_dataset, test_dl = load_data(args, args.test_batch_size, is_train=False)
test_dl = cycle(test_dl)

# Memory
clear_folder(os.path.join(args.memory_path, 'online/'))
memory_dataset = SAC_Jellyfish_pob(vx_min = train_dataset.vx_min,
                            vx_max = train_dataset.vx_max,
                            vy_min = train_dataset.vy_min,
                            vy_max = train_dataset.vy_max,
                            p_min = train_dataset.p_min,
                            p_max = train_dataset.p_max,
                            offline_dir = args.memory_path, 
                            online_dir = os.path.join(args.memory_path, 'online/'),
                            )
memory_dl = DataLoader(memory_dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 16)
memory = cycle(memory_dl)

ppl = build_ppl_new(args)

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

# Training Loop
updates = 0
mask_batch = np.ones((1, num_t))
mask_batch[:, -1] = 0
num_online_data = 0
warm_up_episodes = args.warm_up_episodes

for i_episode in tqdm(range(args.num_episode)):

    t1 = datetime.now()

    if args.online:
        if i_episode >= warm_up_episodes:
            data = next(train_dl)       
            state, mask_offsets, thetas, bd_mask_offset_0, sim_id, time_id = data
            thetas = thetas.to(device)
            state_pad = torch.zeros(state.shape[0], args.frames, 3, s, s).to(device)
            mask_offsets_pad = torch.zeros(mask_offsets.shape[0], args.frames, 3, s, s).to(device)
            bd_mask_offset_0_pad = torch.zeros(bd_mask_offset_0.shape[0], 3, s, s).to(device)
            state_pad = state.to(device)
            mask_offsets_pad[:,:,:,1:-1,1:-1] = mask_offsets.to(device)
            bd_mask_offset_0_pad[:,:,1:-1,1:-1] = bd_mask_offset_0.to(device)
            theta_t, theta_T = thetas[:,0], thetas[:,-1]
            cond_theta = theta_t
            state_t = torch.zeros((state.shape[0], 3, s, s), device=device)
            state_t[:,-1] = state_pad[:,0,-1,:,:]
            mask_offsets_t = mask_offsets_pad[:,0,:,:,:]

            state_all = []
            theta_all = []
            thetas_delta_all = []
            force_all = []
            time_all = []
            next_state_all = []
            mask_all = []
            cond_T = []
            cond_theta_all = []
            for t in range(num_t):
                cond_theta_all.append(cond_theta.cpu().numpy())
                state = torch.cat((state_t, mask_offsets_t), dim=1)
                state = torch.cat((state, mask_offsets_pad[:,0]), 1)
                state_all.append(state.detach().cpu().numpy()) # s0, bd0;...;s19, bd19
                theta_current = theta_t # theta0;...;theta19
                state_time = torch.cat((memory_dataset.normalize_batch(state)[:,2:6], theta_current.reshape(-1,1,1,1).repeat(1,1,s,s),\
                                        cond_theta.reshape(-1,1,1,1).repeat(1,1,s,s), t*torch.ones_like(state[:,[0]])), dim=1)
                theta_t = torch.tensor(agent.select_action(state_time.cpu()), device=device) # theta1;...;theta20
                theta_all.append(theta_t.cpu().numpy()) # theta1;...;theta20
                thetas_delta_all.append((theta_t-theta_current).cpu().numpy()) # theta1-theta0;...;theta20-theta19
                state_t_, force_t = ppl.run_vis_pressure(state_t[:,[-1]], mask_offsets_t, theta_t-theta_current) # output state of next time step and force of current step, s1, f0;...;s20, f19
                state_t[:,[-1]] = state_t_
                mask_offsets_t = ppl.update_mask_offsets_new(mask_offsets_t, theta_t-theta_current) # output mask_offsets of next time step, bd1;...;bd20
                next_state = torch.cat((state_t, mask_offsets_t), dim=1).detach().cpu().numpy()
                if t != 0:
                    force_all.append((num_t + 1 - t) * force_t.cpu().numpy()) # f1;...;f19
                cond_T.append(((theta_t - cond_theta)**2 + 0.1*(theta_t - cond_theta).abs()).cpu().numpy())
                next_state = np.concatenate((next_state, mask_offsets_pad[:,0].cpu().numpy()), 1)
                next_state_all.append(next_state) # s1, bd1;...;s20, bd20
                mask_all.append(np.repeat(mask_batch[0, t], args.train_batch_size))
                time_all.append(t * np.ones_like(theta_all[-1]))

            force_t = ppl.run_vis_pressure(state_t[:,[-1]], mask_offsets_t) # f20
            force_all.append(force_t.cpu().numpy()) # f20
            
            check_for_nan(state_all, theta_all, force_all, next_state_all, mask_all, s)
            now_state_all = np.stack(state_all).reshape(-1, 9, s, s)
            thetas_all = np.stack(theta_all).flatten()
            thetas_delta_all = np.stack(thetas_delta_all).flatten()
            force_all = np.stack(force_all).flatten()
            time_all = np.stack(time_all).flatten()
            cond_T_all = np.array(cond_T).transpose((1,0)).flatten()
            cond_theta_all = np.array(cond_theta_all).transpose((1,0)).flatten()
            next_state_all = np.stack(next_state_all).reshape(-1, 9, s, s)
            mask_batch_all = np.stack(mask_all).flatten()
            for k in range(now_state_all.shape[0]):
                # print(i*args.memory_batch_size*num_t + k)
                np.savez_compressed(os.path.join(args.memory_path, 'online/{}.npz'.format(num_online_data + k)), time_all=time_all[k], now_state_all=now_state_all[k],\
                                    thetas_all=thetas_all[k], thetas_delta_all=thetas_delta_all[k], force_all=force_all[k],\
                                    cond_theta_all=cond_theta_all[k], cond_T_all = cond_T_all[k], next_state_all=next_state_all[k], mask_batch_all=mask_batch_all[k])
            num_online_data += now_state_all.shape[0]
            # print("num_online_data: ", num_online_data)
            # print(len(os.listdir(os.path.join(args.memory_path, 'online/'))))
            # print(os.listdir(os.path.join(args.memory_path, 'online/')))
        
    t2 = datetime.now()
    # print("Time: {}".format(t2-t1))
    # Number of updates per step in environment
    for i in range(args.updates_per_step):
    # Update parameters of all the networks
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, policy_para = agent.update_parameters(memory, args.batch_size, updates, args.reward_alpha, args.reg_lambda)

    t3 = datetime.now()
    updates += 1    

    if (i_episode + 1) % 2  == 0:
        print("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        logs_txt.append("Episode:{}/{}, critic_1_loss:{:1.6f}, critic_2_loss:{:1.6f}, policy_loss:{:1.6f}, ent_loss:{:1.6f}, alpha:{:1.6f}"\
                .format(i_episode, args.num_episode, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, 'results/2d_pde_sac/SAC_policy_track20_pob_{}_{}_{}'.format(date, args.seed, i_episode))

    if (i_episode + 1) % 2 == 0 and args.eval == True:
        avg_reward = 0.
        avg_reward_f = 0.
        avg_reward_bd = 0.
        avg_reg = 0.
        test_batch = 0
        data = next(test_dl)
        state, mask_offsets, thetas, bd_mask_offset_0, sim_id, time_id = data
        thetas = thetas.to(device)
        state_pad = torch.zeros(state.shape[0], args.frames, 3, s, s).to(device)
        mask_offsets_pad = torch.zeros(mask_offsets.shape[0], args.frames, 3, s, s).to(device)
        bd_mask_offset_0_pad = torch.zeros(bd_mask_offset_0.shape[0], 3, s, s).to(device)
        state_pad = state.to(device)
        mask_offsets_pad[:,:,:,1:-1,1:-1] = mask_offsets.to(device)
        bd_mask_offset_0_pad[:,:,1:-1,1:-1] = bd_mask_offset_0.to(device)
        theta_t, theta_T = thetas[:,0], thetas[:,-1]
        cond_theta = theta_t
        state_t = torch.zeros((state.shape[0], 3, s, s), device=device)
        state_t[:,-1] = state_pad[:,0,-1,:,:]
        mask_offsets_t = mask_offsets_pad[:,0,:,:,:]

        force_all = []
        reg_all = []
        theta_test = []
        theta_test.append(theta_t.cpu().numpy())
        for t in range(num_t):
            state = torch.cat((state_t, mask_offsets_t), dim=1)
            state = torch.cat((state, mask_offsets_pad[:,0]), 1)
            theta_current = theta_t
            state_time = torch.cat((memory_dataset.normalize_batch(state)[:,2:6], theta_current.reshape(-1,1,1,1).repeat(1,1,s,s), \
                                    cond_theta.reshape(-1,1,1,1).repeat(1,1,s,s), t*torch.ones_like(state[:,[0]])), dim=1)
            theta_t = torch.tensor(agent.select_action(state_time.cpu(), eval=True), device=device) # theta1;...;theta20
            theta_test.append(theta_t.cpu().numpy())
            reg_all.append(((theta_t-theta_current)**2).cpu().numpy())
            state_t_, force_t = ppl.run_vis_pressure(state_t[:,[-1]], mask_offsets_t, theta_t-theta_current) # output state of next time step and force of current step, s1, f0;...;s20, f19
            state_t[:,[-1]] = state_t_
            mask_offsets_t = ppl.update_mask_offsets_new(mask_offsets_t, theta_t-theta_current) # output mask_offsets of next time step, bd1;...;bd20
            next_state = torch.cat((state_t, mask_offsets_t), dim=1).detach().cpu().numpy()
            if t != 0:
                force_all.append((num_t + 1 - t) * force_t.cpu().numpy()) # f1;...;f19
            avg_reward_bd += ((theta_t - cond_theta)**2 + 0.1*(theta_t - cond_theta).abs()).sum().detach().cpu().numpy() / args.test_batch_size
            next_state = np.concatenate((next_state, mask_offsets_pad[:,0].cpu().numpy()), 1)

        force_t = ppl.run_vis_pressure(state_t[:,[-1]], mask_offsets_t) # f20
        force_all.append((force_t).detach().cpu().numpy()) # f20

        avg_reward_f += np.array(force_all).sum() / args.test_batch_size
        avg_reg += np.array(reg_all).sum() / args.test_batch_size
        avg_reward += args.reward_alpha * avg_reward_f + (-1) * avg_reward_bd + 20 * args.reward_alpha * args.reg_lambda * avg_reg
        
        avg_reward /= args.test_batch
        avg_reward_f /= args.test_batch
        avg_reward_bd /= args.test_batch
        # print(np.stack(theta_test)[:, 0])
        # print(np.stack(theta_test)[:, 10])

        print("Objective_function: {:1.6f}={:1.6f}+{:1.6f}+{:1.6f}".format(avg_reward, avg_reward_f, avg_reward_bd, avg_reg))
        logs_txt.append("Objective_function: {:1.6f}={:1.6f}+{:1.6f}+{:1.6f}".format(avg_reward, avg_reward_f, avg_reward_bd, avg_reg))
        
        logs['critic_1_loss'].append(critic_1_loss)
        logs['critic_2_loss'].append(critic_2_loss)
        logs['policy_loss'].append(policy_loss)
        logs['ent_loss'].append(ent_loss)
        logs['alpha'].append(alpha)
        logs['avg_reward'].append(-avg_reward)
        
        torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, 'results/2d_pde_sac/SAC_policy_track20_pob_{}_{}_{}'.format(date, args.seed, i_episode))
        with open('results/2d_pde_sac/SAC_policy_track20_pob_{}_{}.txt'.format(date, args.seed), 'w') as f:
            for item in logs_txt:
                f.write("%s\n" % item)


torch.save({'logs':logs, 'policy_state_dict': policy_para, 'args':args}, 'results/2d_pde_sac/SAC_policy_track20_pob_{}_{}_{}'.format(date, args.seed, i_episode))
