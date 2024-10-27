import os
import datetime
from datetime import datetime
from scripts.utils import *
from scripts.replay_memory import ReplayMemory
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
from buffer import OfflineReplayBuffer
from critic import ValueLearner, QPiLearner, QSarsaLearner
from bppo import BehaviorCloning, BehaviorProximalPolicyOptimization
from sim_ppl_2d import *
from data_2d import *

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument("--env", default="2D Lilypad")
    parser.add_argument("--seed", default=1100, type=int)
    parser.add_argument("--path", default="logs", type=str)
    parser.add_argument('--dataset_path', default="path", type=str, help='path to dataset') 
    parser.add_argument('--memory_path', default="data/memory_delta_t_20/", type=str, help='path to memory') 
    parser.add_argument('--only_vis_pressure', action='store_true', help="whether only observe pressure, only used in simulator model")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--time_steps', type=int, default=40, help='number of all time steps in each simulation')
    parser.add_argument('--frames', type=int, default=20, help='number of time steps in each training sample')
    parser.add_argument('--m', type=int, default=20, help='number of points on each ellipse wing of jellyfish')
    parser.add_argument('--force_model_checkpoint', type=str, default='path')
    parser.add_argument('--simulator_model_checkpoint', type=str, default='path')
    parser.add_argument('--boundary_updater_model_checkpoint', type=str, default='path')
    parser.add_argument('--is_testdata', default=False, type=bool,
                    help='whether run mini example data, if True, yes; otherwise, run full data')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
    parser.add_argument('--memory_batch_size', type=int, default=2800, metavar='N',
                        help='batch size for memory pushing (default: 2800)')
    parser.add_argument("--online", action='store_true', help='online?')
    parser.add_argument('--reward_alpha', type=float, default=0.0001, metavar='G',
                        help='weight of limitation of theta_T in reward')
    parser.add_argument('--reg_lambda', type=float, default=-1000, metavar='G',
                        help='weight of regularizer of theta')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Deterministic)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 1)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
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
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
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
    parser.add_argument("--log_freq", default=int(2e1), type=int)
    
    # For Value
    parser.add_argument("--v_steps", default=int(2e1), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(2e1), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=1024, type=int)
    parser.add_argument("--q_depth", default=2, type=int)       
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e1), type=int)
    parser.add_argument("--bc_hidden_dim", default=1024, type=int)
    parser.add_argument("--bc_depth", default=2, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    # For BPPO 
    parser.add_argument("--bppo_steps", default=int(1e1), type=int)
    parser.add_argument("--bppo_hidden_dim", default=1024, type=int)
    parser.add_argument("--bppo_depth", default=2, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0, type=float) 
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.9, type=float)
    parser.add_argument("--is_clip_decay", default=True, type=bool)  
    parser.add_argument("--is_bppo_lr_decay", default=True, type=bool)       
    parser.add_argument("--is_update_old_policy", default=True, type=bool)
    parser.add_argument("--is_state_norm", default=False, type=bool)
    parser.add_argument("--channel", default=10, type=int)
    
    def rel_error(x, _x):
        """
        <ARGS>
        x : torch.Tensor shape of (B, *)
        _x : torch.Tensor shape of (B, *)
        <RETURN>
        out :torch.Tensor shape of (B), batchwise relative error between x and _x : (||x-_x||_2/||_x||_2)
        
        """
        if len(x.shape)==1:
            x = x.reshape(1, -1)
            _x = _x.reshape(1, -1)
        else:
            B = x.size(0)
            x, _x = x.reshape(B, -1), _x.reshape(B, -1)
        return torch.norm(x - _x, 2, dim=1)/torch.norm(_x, 2, dim=1)
    
    args = parser.parse_args()
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    os.makedirs(os.path.join(path, current_time))
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    num_t=args.frames - 1
    s = args.image_size
    
    # Online data
    train_dataset, train_dl = load_data(args, args.train_batch_size, is_train=True)
    train_dl = cycle(train_dl)

    # Test data
    test_dataset, test_dl = load_data(args, args.test_batch_size, is_train=False)
    test_dl = cycle(test_dl)
    
    # Memory
    clear_folder(os.path.join(args.memory_path, 'online/'))
    memory_dataset = SAC_Jellyfish(vx_min = train_dataset.vx_min,
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
    
    state_dim = s
    action_dim = 1

    # summarywriter logger
    comment = args.env + '_' + str(args.seed)
    logger_path = os.path.join(path, current_time)    
        
    # initilize
    value = ValueLearner(device, state_dim, args.v_lr, args.v_batch_size, args.channel)
    Q_bc = QSarsaLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size, args.channel)
    bc = BehaviorCloning(device, state_dim, args.bc_hidden_dim, args.bc_depth, action_dim, args.bc_lr, args.bc_batch_size, args.channel)
    bppo = BehaviorProximalPolicyOptimization(device, state_dim, args.bppo_hidden_dim, args.bppo_depth, action_dim, args.bppo_lr, args.clip_ratio, args.entropy_weight, args.decay, args.omega, args.bppo_batch_size, args.channel)


    # value training 
    value_path = os.path.join(path, 'value.pt')
    if os.path.exists(value_path):
        value.load(value_path)
    else:
        for step in tqdm(range(int(args.v_steps)), desc='value updating ......'): 
            value_loss = value.update(memory, args.reward_alpha, args.reg_lambda)
                        
            if step % int(args.log_freq) == 0:
                print(f"Step: {step}, Loss: {value_loss:.4f}")

        value.save(value_path)

    # Q_bc training
    Q_bc_path = os.path.join(path, 'Q_bc.pt')
    if os.path.exists(Q_bc_path):
        Q_bc.load(Q_bc_path)
    else:
        for step in tqdm(range(int(args.q_bc_steps)), desc='Q_bc updating ......'): 
            Q_bc_loss = Q_bc.update(memory, args.reward_alpha, args.reg_lambda)
                        
            step_qbc = step

            if step % int(args.log_freq) == 0:
                print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")

        Q_bc.save(Q_bc_path)
    

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
    
    # bppo training    
    bppo.load(best_bc_path)
    best_bppo_path = os.path.join(path, current_time, 'bppo_best.pt')
    Q = Q_bc

    for step in tqdm(range(int(args.bppo_steps)), desc='bppo updating ......'):
        if step > 200:
            args.is_clip_decay = False
            args.is_bppo_lr_decay = False
        bppo_loss = bppo.update(memory, Q, value, args.is_clip_decay, args.is_bppo_lr_decay)
        
        bppo.save(f'{best_bppo_path}_{step}')

        print(f"Step: {step}, Loss: {bppo_loss:.4f}")
        
                
    #################### evaluate ###########
    
    def select_action(s: torch.Tensor, is_sample: bool):
        dist = bppo._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.mean
        # clip 
        action = action.clamp(-1., 1.)
        
        return action[:,0]
    
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
    state_t = state_pad[:,0,:,:,:]
    mask_offsets_t = mask_offsets_pad[:,0,:,:,:]

    force_all = []
    reg_all = []
    theta_test = []
    theta_test.append(theta_t.cpu().numpy())
    
    for t in range(num_t):
        state = torch.cat((state_t, mask_offsets_t), dim=1)
        state = torch.cat((state, mask_offsets_pad[:,0]), 1)
        theta_current = theta_t
        state_time = torch.cat((memory_dataset.normalize_batch(state)[:,:6], theta_current.reshape(-1,1,1,1).repeat(1,1,s,s), \
                                cond_theta.reshape(-1,1,1,1).repeat(1,1,s,s), t*torch.ones_like(state[:,[0]])), dim=1)
        
        
        theta_t = torch.tensor(select_action(state_time, is_sample=False)) # theta1;...;theta20
        
        theta_test.append(theta_t.cpu().numpy())
        reg_all.append(((theta_t-theta_current)**2).cpu().numpy())
        state_t, force_t = ppl.run(state_t, mask_offsets_t, theta_t-theta_current) # output state of next time step and force of current step, s1, f0;...;s20, f19
        mask_offsets_t = ppl.update_mask_offsets_new(mask_offsets_t, theta_t-theta_current) # output mask_offsets of next time step, bd1;...;bd20
        next_state = torch.cat((state_t, mask_offsets_t), dim=1).detach().cpu().numpy()
        if t != 0:
            force_all.append((num_t + 1 - t) * force_t.cpu().numpy()) # f1;...;f19
        avg_reward_bd += ((theta_t - cond_theta)**2 + 0.1*(theta_t - cond_theta).abs()).sum().detach().cpu().numpy() / args.test_batch_size
        next_state = np.concatenate((next_state, mask_offsets_pad[:,0].cpu().numpy()), 1)

    force_t = ppl.run(state_t, mask_offsets_t) # f20
    force_all.append((force_t).detach().cpu().numpy()) # f20

    avg_reward_f += np.array(force_all).sum() / args.test_batch_size
    avg_reg += np.array(reg_all).sum() / args.test_batch_size
    avg_reward += args.reward_alpha * avg_reward_f + (-1) * avg_reward_bd + 20 * args.reward_alpha * args.reg_lambda * avg_reg
    
    avg_reward /= args.test_batch
    avg_reward_f /= args.test_batch
    avg_reward_bd /= args.test_batch

    thetas1 = np.stack(theta_test)  # (t,bz)

    print("Objective_function: {:1.6f}={:1.6f}+{:1.6f}+{:1.6f}".format(avg_reward, avg_reward_f, avg_reward_bd, avg_reg))
    logs_txt.append("Objective_function: {:1.6f}={:1.6f}+{:1.6f}+{:1.6f}".format(avg_reward, avg_reward_f, avg_reward_bd, avg_reg))
