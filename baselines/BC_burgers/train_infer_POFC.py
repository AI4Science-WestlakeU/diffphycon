import os
import datetime
import h5py
from datetime import datetime
from scripts.utils import *
from scripts.replay_memory import ReplayMemory
from generate_burgers import burgers_numeric_solve_free, burgers_numeric_solve
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
import wandb
from buffer import OfflineReplayBuffer
from critic import ValueLearner, QPiLearner, QSarsaLearner
from bppo import BehaviorCloning, BehaviorProximalPolicyOptimization
from burgers_operator import Simu_surrogate_model

def store(path, id, u):
    np.save(f'{path}/{id}',u)

def cycle(dl):
    while True:
        for data in dl:
            yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="1D burgers")        
    parser.add_argument("--seed", default=137, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(2e3), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(2e6), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(2e6), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=1024, type=int)
    parser.add_argument("--q_depth", default=2, type=int)       
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e5), type=int) # try to reduce the bc/q/v step if it works poorly, 5e-4/2e-5/2e-5 for bc/q/v, for example
    parser.add_argument("--bc_hidden_dim", default=1024, type=int)
    parser.add_argument("--bc_depth", default=2, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    # For BPPO 
    parser.add_argument("--bppo_steps", default=int(1e1), type=int)
    parser.add_argument("--bppo_hidden_dim", default=1024, type=int)
    parser.add_argument("--bppo_depth", default=2, type=int)
    parser.add_argument("--bppo_lr", default=1e-5, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0, type=float) # for ()-medium-() tasks, try to use the entropy loss, weight == 0.01
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.9, type=float)
    parser.add_argument("--is_clip_decay", default=True, type=bool)  
    parser.add_argument("--is_bppo_lr_decay", default=True, type=bool)       
    parser.add_argument("--is_update_old_policy", default=True, type=bool)
    parser.add_argument("--is_state_norm", default=False, type=bool)
    
    parser.add_argument('--reward_f', type=float, default=1e-5, metavar='G',
                    help='weight of energy')
    parser.add_argument('--replay_size', type=int, default=2000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
    parser.add_argument('--grid_size', default=128, type=int, 
                    help = 'grid size (default: 128)')
    parser.add_argument('--train_batch_size', type=int, default=1, metavar='N',
                    help='training batch size (default: 5)')
    parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                        help='test batch size (default: 256)')
    parser.add_argument('--lf', type=str, default='BPPO_origin', metavar='N',
                        help='test batch size (default: 256)')
    parser.add_argument('--is_relative', type=int, default=0, metavar='N',
                    help='reward calculated by relative or mse')
    parser.add_argument('--pde_1d_surrogate_model_checkpoint', type=str, default='surrogate_model/POFC/', metavar='N',
                        help='test batch size (default: 256)')
    
    
    def rel_error(x, _x):
        if len(x.shape)==1:
            x = x.reshape(1, -1)
            _x = _x.reshape(1, -1)
        else:
            B = x.size(0)
            x, _x = x.reshape(B, -1), _x.reshape(B, -1)
        return torch.norm(x - _x, 2, dim=1)/torch.norm(_x, 2, dim=1)
    
    args = parser.parse_args()
    
    RESCALER = 10.
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    reward_f = args.reward_f
    num_t=10
    s = args.grid_size
    memory = ReplayMemory(args.replay_size)
    mask_batch = np.zeros((num_t))
    dataset = h5py.File("../../1D_data/free_u_f_1e5/burgers_train.h5", 'r')['train']
    u_data, f_data = (torch.tensor(np.array(dataset['pde_11-128']), device=device).float())/RESCALER, (torch.tensor(np.array(dataset['pde_11-128_f']), device=device).float())/RESCALER
    u_data_full = u_data
    
    u_data = torch.cat((u_data[:, :, :int(s/4)], torch.zeros_like(u_data[:,:,int(s/4):int(3*s/4)]), u_data[:, :, int(3*s/4):]), -1)
    Ndata = u_data.shape[0]
    u_target = u_data[:, -1]
    
    for i in range(num_t):
        state = torch.cat((i*torch.ones_like(u_data[:, 0, [0]])/RESCALER, u_data[:, i], u_target), -1)  
        action = f_data[:, i]  
        
        if args.is_relative == 1:
            uT_reward = (-1) * rel_error(u_data[:, i+1], u_target)
        elif args.is_relative == 0:
            uT_reward = (-1) * (u_target-u_data[:, i+1]).square().mean(-1)

        f_reward = torch.sum(f_data[:, i]**2, -1) 
        reward = uT_reward + (-reward_f)*f_reward
        
        next_state = torch.cat(((i+1)*torch.ones_like(u_data[:, 0, [0]])/RESCALER, u_data[:, i+1], u_target), -1)

        memory.push_batch(list(zip(state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(),\
                        next_state.cpu().numpy(), np.repeat(mask_batch[i], Ndata))))

    train_dl = DataLoader(TensorDataset(torch.cat((u_data_full.cpu(), f_data.cpu()), 1)), batch_size = args.train_batch_size, shuffle = True)
    train_dl = cycle(train_dl)
    
    state_dim = int((s)*2+1)
    action_dim = s

    comment = args.env + '_' + str(args.seed)
    logger_path = os.path.join(path, current_time)    
    
    value = ValueLearner(device, state_dim, args.v_hidden_dim, args.v_depth, args.v_lr, args.v_batch_size)
    Q_bc = QSarsaLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size)
    if args.is_offpolicy_update:
        Q_pi = QPiLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size)
    bc = BehaviorCloning(device, state_dim, args.bc_hidden_dim, args.bc_depth, action_dim, args.bc_lr, args.bc_batch_size)
    bppo = BehaviorProximalPolicyOptimization(device, state_dim, args.bppo_hidden_dim, args.bppo_depth, action_dim, args.bppo_lr, args.clip_ratio, args.entropy_weight, args.decay, args.omega, args.bppo_batch_size)

    value_path = os.path.join(path, 'value.pt')
    if os.path.exists(value_path):
        value.load(value_path)
    else:
        for step in tqdm(range(int(args.v_steps)), desc='value updating ......'): 
            value_loss = value.update(memory)
            if step % int(args.log_freq) == 0:
                print(f"Step: {step}, Loss: {value_loss:.4f}")

        value.save(value_path)

    # Q_bc training
    Q_bc_path = os.path.join(path, 'Q_bc.pt')
    if os.path.exists(Q_bc_path):
        Q_bc.load(Q_bc_path)
    else:
        for step in tqdm(range(int(args.q_bc_steps)), desc='Q_bc updating ......'): 
            Q_bc_loss = Q_bc.update(memory, pi=None)     
            step_qbc = step

            if step % int(args.log_freq) == 0:
                print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")

        Q_bc.save(Q_bc_path)
        
    
    if args.is_offpolicy_update:
        Q_pi.load(Q_bc_path)
    

    # bc training
    best_bc_path = os.path.join(path, 'bc_best.pt')
    if os.path.exists(best_bc_path):
        bc.load(best_bc_path)
    else:
        best_bc_score = 0    
        for step in tqdm(range(int(args.bc_steps)), desc='bc updating ......'):
            bc_loss = bc.update(memory)

            if step % int(args.log_freq) == 0:
                bc.save(best_bc_path)
                np.savetxt(os.path.join(path, 'best_bc.csv'), [best_bc_score], fmt='%f', delimiter=',')
                print(f"Step: {step}, Loss: {bc_loss:.4f}")

        bc.save(os.path.join(path, 'bc_last.pt'))
        bc.load(best_bc_path)

    bppo.load(best_bc_path)
        
    #################### evaluate ###########
    
    logs = dict()
    logs['states'] = []
    logs['target'] = []
    logs['ddpm_mse'] = []
    logs['J_diffused'] = []
    logs['J_actual_mse'] = []
    logs['J_actual_nmae'] = []
    logs['J_relative'] = []
    logs['energy'] = []  
    logs['f'] = []  
    logs['J_actual_rl2'] = []
    
    def evaluate(seed, eval_episodes, s_0, test_target, test_u_full):
        total_reward = torch.zeros(s_0.shape[0], eval_episodes)
        total_rl2 = torch.zeros(s_0.shape[0], eval_episodes)
        total_f = torch.zeros(s_0.shape[0], eval_episodes)
        u_sum = torch.zeros(50,11,128)
        f_sum = torch.zeros(50,11,128)
        for j in range(eval_episodes):
                        
            if j==0:
                state_current = torch.FloatTensor(s_0).to(device)
                state_full = test_u_full[:,0].to(device)
            else:
                state_current = torch.FloatTensor(state_current).to(device)
                state_full = (output_solver/RESCALER).to(device)
                
            target = state_current[:,-128:]*RESCALER  
            
            a = select_action(state_current, is_sample=False)

            in_surr_model = torch.cat((state_full[:, :int(s/4)], state_full[:, int(3*s/4):]), -1)
            
            f_sum[:,j] = a*RESCALER

            s_feedback = simu_surrogate_model.simulation(ut=in_surr_model*RESCALER,ft=a.unsqueeze(1)*RESCALER)
            
            in_solver = torch.cat((state_full[:, :int(s/4)], torch.zeros_like(state_full[:,int(s/4):int(3*s/4)]), state_full[:, int(3*s/4):]), -1)
            s_feedback_solver = burgers_numeric_solve_free(in_solver*RESCALER, a.unsqueeze(1)*RESCALER, visc=0.01, T=0.1, dt=1e-4, num_t=1)
            
            if j==0:
                u_sum[:,0] = in_solver*RESCALER
                
            output_solver = s_feedback[:,0]   
            
            u_sum[:,j+1] = s_feedback_solver[:,0]
            
            output_solver = torch.cat((output_solver[:, :int(s/4)], torch.zeros_like(output_solver[:,:]), output_solver[:, int(s/4):]), -1)
            
            loss_cal_u = torch.cat((s_feedback_solver[:,0][:, :int(s/4)], s_feedback_solver[:,0][:, int(3*s/4):]), -1)
            loss_cal_target = torch.cat((target[:, :int(s/4)], target[:, int(3*s/4):]), -1)

            r = (loss_cal_u-loss_cal_target).square().mean(-1)
            rl2 = rel_error(loss_cal_u, loss_cal_target)
            
            state_current = torch.cat(((j+1)*torch.ones_like(test_target[:,[0]])/RESCALER, (output_solver.cpu())/RESCALER, test_target), -1)

            f_cost = torch.sum(a**2, -1) 

            total_reward[:,j] = r
            total_rl2[:,j] = rl2
            total_f[:,j] = f_cost
                    
        logs['J_actual_mse'] = total_reward
        logs['J_actual_rl2'] = total_rl2 
        logs['f'] = total_f
        
        return total_reward.mean()
    
    def select_action(s: torch.Tensor, is_sample: bool):
        dist = bppo._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.mean
        action = action.clamp(-1., 1.)
        return action
    
    simu_surrogate_model = Simu_surrogate_model(path=args.pde_1d_surrogate_model_checkpoint,device=device,s_ob=64)
    # Test data
    test_dataset = h5py.File("../../1D_data/free_u_f_1e5/burgers_test.h5", 'r')['test']
    
    test_u_data, test_f_data = ((torch.tensor(np.array(test_dataset['pde_11-128'])).float())/RESCALER)[:50], ((torch.tensor(np.array(test_dataset['pde_11-128_f'])).float())/RESCALER)[:50]
    
    test_u_full = test_u_data
    test_u_data = torch.cat((test_u_data[:, :, :int(s/4)], torch.zeros_like(test_u_data[:,:,int(s/4):int(3*s/4)]), test_u_data[:, :, int(3*s/4):]), -1)

    test_target = test_u_data[:,-1]
    
    state = torch.cat((0*torch.ones_like(test_u_data[:, 0, [0]])/RESCALER, test_u_data[:, 0], test_target), -1)  
    avg_reward = evaluate(args.seed, 10, state, test_target, test_u_full)
        
    torch.save(logs, f'../baseline_eval_result/{args.lf}')
    

    
