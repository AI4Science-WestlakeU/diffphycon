      
import argparse
from collections import OrderedDict
import datetime
import matplotlib.pylab as plt
from numbers import Number
import numpy as np
import math as math_package
import pdb
import pickle
import pprint as pp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import grad
import tqdm
from accelerate import Accelerator
from pathlib import Path
import gc
from ddpm.data_2d import Smoke, Smoke_wave
from ddpm.diffusion_2d import Unet, GaussianDiffusion, Trainer
from ddpm.wave_utils import upsample_coef
from ddpm.utils import replace_nan
from dataset.apps.evaluate_solver import *
import sys, os
from matplotlib.backends.backend_pdf import PdfPages
import train_surrogate_models_2d
from ddpm.diffusion_2d import Unet, Simulator
from bppo import BehaviorCloning, BehaviorProximalPolicyOptimization
from IPython import embed
import multiprocess as mp

def clip_loss(tensor, min_value, max_value):
    clipped_tensor = torch.clamp(tensor, min_value, max_value)
    
    loss = (tensor - clipped_tensor).abs().mean()
    
    return loss

from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
def get_CosineAnnealingLR(num_iters,lr_max=0.0001,eta_min=0.0000001):
    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=lr_max)
    total_epochs = num_iters
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    learning_rates = []
    coef_grad_list=torch.zeros(num_iters)
    for epoch in range(total_epochs):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        coef_grad_list[epoch]=current_lr
    return coef_grad_list

def load_model(args, shape, ori_shape, RESCALER, w_energy=0, w_init=0):
    RESCALER = RESCALER
    # load main model of each inference method
    best_bc_path = args.bppo_model_path
    model = BehaviorProximalPolicyOptimization(args.device, args.image_size, args.bppo_hidden_dim, args.bppo_depth, 1, args.bppo_lr, args.clip_ratio, args.entropy_weight, args.decay, args.omega, args.bppo_batch_size, args.channel)
    model.load(best_bc_path)
    return model, design_fn
     
import matplotlib.pyplot as plt
class InferencePipeline(object):
    def __init__(
        self,
        model,
        args=None,
        RESCALER=1,
        results_path=None,
        args_general=None,
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.results_path = results_path
        self.args_general = args_general
        self.is_wavelet = args_general.is_wavelet
        self.is_condition_control = args_general.is_condition_control
        self.image_size = self.args_general.image_size
        self.device = self.args_general.device
        self.upsample = args_general.upsample
        self.RESCALER = RESCALER
        self.simulator = Simulator(
                                    dim = args_general.image_size,
                                    out_dim = 4,
                                    dim_mults = (1, 2, 4),
                                    channels = 6
        )
        self.simulator.to(self.device)
        self.simulator.load_state_dict(torch.load(args_general.surrogate_model_path))
    
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def run_model_BPPO(self, state):
        def select_action(s: torch.Tensor, is_sample: bool):
            dist = self.model._policy(s)
            if is_sample:
                action = dist.sample()
            else:    
                action = dist.mean
            # clip 
            action = action.clamp(-1., 1.)
            
            return action
        
        def memory_usage(tensor):
            num_elements = tensor.numel()
            element_size = tensor.element_size()
            memory_in_bytes = num_elements * element_size
            memory_in_megabytes = memory_in_bytes / (1024 ** 2)
            return memory_in_megabytes
        
        def model_memory_usage(model):
            total_memory = 0
            for param in model.parameters():
                num_elements = param.numel()
                element_size = param.element_size()
                memory_in_bytes = num_elements * element_size
                total_memory += memory_in_bytes
            
            total_memory_in_mb = total_memory / (1024 ** 2)
            return total_memory_in_mb
        
        def clear_model_memory(model):
            for param in model.parameters():
                param.grad = None
            for name, param in model.named_parameters():
                del param
            del model
            gc.collect()
            torch.cuda.empty_cache()
        
        s = self.image_size
        num_t = 31

        state = state[:,::8].to(self.device) / self.RESCALER.to(self.device)
        state_t=state[:,0] 
        
        time = torch.zeros(state.shape[0], device=self.device).reshape(-1, 1, 1, 1).expand(-1, -1, s, s)
        action = torch.tensor(select_action(torch.cat((state_t[:,:3],time/(num_t)),dim=1), is_sample=False)) 
        
        action[:,:,8:56,8:56] = 0
        state_t[:,3:5]=action
        pred = torch.zeros_like(state)
        pred[:,0,0] = state_t[:,0]
        pred[:,0,-1] = state_t[:,-1]
        pred[:,0,3:5] = action
        
        self.simulator.eval()
        
        for t in range(1, num_t+1):            
            
            state_t = self.simulator(state_t).data 
            if t != num_t:
                time = t*torch.ones(state.shape[0], device=self.device).reshape(-1, 1, 1, 1).expand(-1, -1, s, s)
                action = torch.tensor(select_action(torch.cat((state_t[:,:3],time/num_t),dim=1), is_sample=False))  
                action[:,:,8:56,8:56] = 0
                state_t = torch.cat((state_t[:,:3], action, state_t[:,[-1]]), dim=1).detach()
                pred[:,t,1:3] = state_t[:,1:3]
                pred[:,t,0] = state_t[:,0].detach()
                pred[:,t,-1] = state_t[:,-1].detach()
                pred[:,t,3:5] = action.detach()

        return pred * self.RESCALER

    
    def run_model(self, state):
        '''
        state: not rescaled
        '''
        state_ori = state.to(self.args_general.device)
        if not self.args_general.is_condition_control:
            state = state_ori[:, ::8] 
        else:
            state = state_ori[:, :, :, ::2, ::2] 
        
        if len(self.model) == 1:
            output = self.model[0].sample(
                batch_size = state.shape[0],
                design_fn=self.args["design_fn"],
                design_guidance=self.args["design_guidance"],
                low=None, 
                init=wave_init/self.RESCALER[:,:,-2] if self.is_wavelet \
                        else state[:,0,0]/self.RESCALER[:,0,0], 
                init_u=state[:,0,0], 
                control=wave_control/self.RESCALER[:,:,24:40] if self.is_wavelet \
                        else state[:,:,3:5]/self.RESCALER[:,:,3:5]
                )
            if not args.is_wavelet:
                output = output * self.RESCALER
                output[:,:,-1] = output[:,:,-1].mean((-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,64,64)
                return output
            else: 
                shape, ori_shape = self.model[0].padded_shape, self.model[0].ori_shape
                wave_output = output * self.RESCALER
                coef = tensor_to_coef(wave_output[:,:,:-2].permute(0,2,1,3,4), shape)
                ori_output = ptwt.waverec3(coef, pywt.Wavelet(self.model[0].wave_type))\
                            [:, :ori_shape[0], :ori_shape[1], :ori_shape[2]]\
                            .reshape(-1, 5, ori_shape[0], ori_shape[1], ori_shape[2]).permute(0,2,1,3,4)
                
                ifm1d = DWT1DInverse(mode=self.model[0].pad_mode, wave=self.model[0].wave_type).to(output.device)
                Yl_s = wave_output[:,:shape[0],-1,:int(40/2)].mean((-2,-1)).unsqueeze(1)
                Yh_s = [wave_output[:,:shape[0],-1,int(40/2):].mean((-2,-1)).unsqueeze(1)]
                smoke_out = ifm1d((Yl_s, Yh_s))[:,0] # 25, 32
                smoke_out = smoke_out.reshape(smoke_out.shape[0], smoke_out.shape[1], 1, 1, 1)\
                            .expand(-1, -1, -1, 64, 64)
                ori_output = torch.cat((ori_output, smoke_out), dim=2)
                return ori_output            

    
    def run(self, dataloader):
        preds = []
        J_totals, J_targets, J_energys, mses, n_l2s = {}, {}, {}, {}, {}
        for i in range(self.upsample+1):
            J_totals[i], J_targets[i], J_energys[i], mses[i], n_l2s[i] = [], [], [], [], []
        for i, data in enumerate(dataloader):
            print(f"Batch No.{i}")
            state, shape, ori_shape, sim_id = data 
            pred = self.run_model_BPPO(state)
            preds.append(pred)
            if not self.args_general.is_super_model:
                J_total, J_target, J_energy, mse, n_l2 = self.multi_evaluate(pred, state, plot=False, method=self.args_general.inference_method)
                J_totals[0].append(J_total)
                J_targets[0].append(J_target)
                J_energys[0].append(J_energy)
                mses[0].append(mse)
                n_l2s[0].append(n_l2)
            else:
                for i in range(len(pred)):
                    print('Number of upsampling times:', i)
                    J_total, J_target, J_energy, mse, n_l2 = self.multi_evaluate(pred[i], state, plot=False)
                    J_totals[i].append(J_total)
                    J_targets[i].append(J_target)
                    J_energys[i].append(J_energy)
                    mses[i].append(mse)
                    n_l2s[i].append(n_l2)
            
            torch.cuda.empty_cache()

        print("Final results!")
        # save
        for i in range(self.upsample+1):
            print(f"Number of upsampling times: {i}")
            print(f"J_total: {np.stack(J_totals[i]).mean(0)},\nJ_target: {np.stack(J_targets[i]).mean(0)},\nJ_energy: {np.stack(J_energys[i]).mean(0)},\nmse: {np.stack(mses[i]).mean(0)},\nn_l2: {np.stack(n_l2s[i]).mean(0)}")
        save_file = 'results_sim.txt' if self.args_general.is_condition_control else 'results.txt'
        with open(os.path.join(self.results_path, save_file), 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\n')
            f.write(str(self.args_general)+'\n')
            for i in range(self.upsample+1):
                f.write(f"Number of upsampling times: {i}\n")
                f.write(f"J_total: {np.stack(J_totals[i]).mean(0)},\nJ_target: {np.stack(J_targets[i]).mean(0)},\nJ_energy: {np.stack(J_energys[i]).mean(0)},\nmse: {np.stack(mses[i]).mean(0)},\nn_l2: {np.stack(n_l2s[i]).mean(0)}\n")
            f.write("-----------------------------------------------------------------------------------------\n")    
    
    def per_evaluate(self, sim, eval_no, pred, data, output_queue):
        '''
        eval_no: No of multi-process
        pred: torch.Tensor, [nt, 1, nx, nx]
        '''
        if not self.args_general.is_condition_control:         
            init_velocity = init_velocity_() 
            init_density = data[0,0,:,:] # nx, nx
            c1 = pred[:,3,:,:] # nt, nx, nx
            c2 = pred[:,4,:,:] # nt, nx, nx
            per_solver_out = solver(sim, init_velocity, init_density, c1, c2, per_timelength=256)

        try:
            output_queue.put({eval_no:per_solver_out})
        except Exception as e:
            print(f"Error in process {eval_no}: {e}")



    def multi_evaluate(self, pred, data, plot=False, method='DDPM'):
        '''
        pred: torch.Tensor, [B, nt, 6, nx, nx] 
        data: torch.Tensor, control: [B, 256, 1, 64, 64], simulation: [B, 32, 1, 128, 128]
        '''
        pred[:, 0, 0] = data[:, 0, 0, ::int(data.shape[-1]/pred.shape[-1]), ::int(data.shape[-1]/pred.shape[-1])] # initial condition
        if not self.args_general.is_condition_control: # control 
            print("Start solving...")
            start = time.time()
            pool_num = pred.shape[0]
            solver_out = np.zeros((pred.shape[0],256,6,128,128), dtype=float)
            pred_ = pred.detach().cpu().numpy().copy()
            data_ = data.detach().cpu().numpy().copy()
            pred_[:,:,3:5,8:56,8:56] = 0
            sim = init_sim_128()
            output_queue = mp.Queue()

            processes = []
            args_list = [(sim, i, pred_[i,:,:,:,:].copy(), data_[i,:,:,:,:].copy(),output_queue) for i in range(pool_num)]

            for args in args_list:
                process = mp.Process(target=self.per_evaluate, args=args)
                processes.append(process)
                process.start()

            multi_results_list = []

            for i in range(len(processes)):
                multi_results_list.append(output_queue.get())
            
            multi_results_sorted = dict()
            for eval_no in range(len(processes)): # process no.
                for item in multi_results_list: 
                    if list(item.keys())[0] == eval_no:
                        multi_results_sorted[f'{eval_no}']=list(item.values())[0]
                        continue

            for process in processes:
                process.join()

            for i in range(len(multi_results_sorted)):
                solver_out[i,:,0,:,:] = multi_results_sorted[f'{i}'][0] # density
                solver_out[i,:,1,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,0] # vel_x
                solver_out[i,:,2,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,1] # vel_x
                solver_out[i,:,3,:,:] = multi_results_sorted[f'{i}'][3] # control_x
                solver_out[i,:,4,:,:] = multi_results_sorted[f'{i}'][4] # control_y
                solver_out[i,:,5,:,:] = multi_results_sorted[f'{i}'][5] # smoke_portion

            if plot:
                print("Start Generating GIFs...")
                """
                Generate GIF
                """
                for i in range(10):
                    gif_control(solver_out[i,:,3,:,:], solver_out[i,:,4,:,:],control_bool=True,name=f'{i}')
                    gif_density(solver_out[i,:,0,:,:],zero=False,name=f'{i}')
                    gif_control(solver_out[i,:,1,:,:], solver_out[i,:,4,:,:],control_bool=False,name=f'{i}')

            data_super = torch.tensor(solver_out, device=pred.device)[:, :, :, ::2, ::2] 
            data_current = data_super[:, ::int(data_super.shape[1]/pred.shape[1])]
            data_base = data_super[:, ::8]
            end = time.time()
            print(f"Time cost: {end-start}")

            pred_current = pred
            if method == 'DDPM' and self.args_general.is_super_model:
                pred_base = pred[:, ::int(pred.shape[1]/32)]
                pred_super = F.interpolate(pred.permute(0,2,3,4,1).reshape(pred.shape[0],-1,pred.shape[1]), size=(256), mode='linear', align_corners=False)\
                            .permute(0,2,1).reshape(pred.shape[0],256,pred.shape[2],pred.shape[3],pred.shape[4])
                pred_super2 = F.interpolate(pred.permute(0,2,3,4,1).reshape(pred.shape[0],-1,pred.shape[1]), size=(256), mode='nearest')\
                            .permute(0,2,1).reshape(pred.shape[0],256,pred.shape[2],pred.shape[3],pred.shape[4])
        else:
            data_super = data.to(pred.device)
            data_current = data_super[:, :, :, ::int(data_super.shape[-1]/pred.shape[-1]), ::int(data_super.shape[-1]/pred.shape[-1])]
            data_base = data_super[:, :, :, ::2, ::2]

            pred_current = pred
            if method == 'DDPM' and self.args_general.is_super_model:
                pred_base = pred[:, :, :, ::int(pred.shape[-1]/64), ::int(pred.shape[-1]/64)]
                pred_super = F.interpolate(pred.reshape(pred.shape[0],-1,*pred.shape[-2:]), size=(128,128), mode='bilinear', align_corners=False)\
                            .reshape(pred.shape[0],pred.shape[1],pred.shape[2],128,128)
                pred_super2 = F.interpolate(pred.reshape(pred.shape[0],-1,*pred.shape[-2:]), size=(128,128), mode='nearest')\
                            .reshape(pred.shape[0],pred.shape[1],pred.shape[2],128,128)

        if method == 'DDPM' and self.args_general.is_super_model:
            data = [data_base, data_current, data_super, data_super]
            pred = [pred_base, pred_current, pred_super, pred_super2]
            print('current resolution:', pred[1].shape[1], pred[1].shape[3], pred[1].shape[4])
        else: 
            data = [data_current]
            pred = [pred_current]
        J_totals, J_targets, J_energys, mses, n_l2s = [], [], [], [], []
        for i in range(len(pred)):
            mask = torch.ones_like(pred[i], device = pred[i].device)
            mask[:, 0, 0] = False
            pred[i] = pred[i] * mask 
            data[i] = data[i] * mask
            mse = (torch.cat(((pred[i] - data[i])[:,:,:3], (pred[i] - data[i])[:,:,[-1]]), dim=2).square().mean((1, 2, 3, 4))).detach().cpu().numpy() # base resolution
            mse_wo_smoke = ((pred[i] - data[i])[:,:,:3].square().mean((1, 2, 3, 4))).detach().cpu().numpy() # base resolution
            n_l2 = (((pred[i] - data[i])[:,:,:3].square().sum((1, 2, 3, 4))).sqrt()/data[i][:,:,:3].square().sum((1, 2, 3, 4)).sqrt()).detach().cpu().numpy()
            n_l2_density = (((pred[i] - data[i])[:,:,0].square().sum((1, 2, 3))).sqrt()/data[i][:,:,0].square().sum((1, 2, 3)).sqrt()).detach().cpu().numpy()
            n_l2_v1 = (((pred[i] - data[i])[:,:,1].square().sum((1, 2, 3))).sqrt()/data[i][:,:,1].square().sum((1, 2, 3)).sqrt()).detach().cpu().numpy()
            n_l2_v2 = (((pred[i] - data[i])[:,:,2].square().sum((1, 2, 3))).sqrt()/data[i][:,:,2].square().sum((1, 2, 3)).sqrt()).detach().cpu().numpy()
            mae_smoke = (pred[i][:,-1,5].mean((1,2)) - data[i][:,-1,5].mean((1,2))).abs().detach().cpu().numpy()
        
            J_target = - data[i][:, -1, -1, 0, 0].detach().cpu().numpy()
            J_energy = data[i][:, :, 3:5].square().mean((1, 2, 3, 4)).detach().cpu().numpy()
            J_total = J_target + self.args_general.w_energy * J_energy
            print('evaluate shape:', pred[i].shape[1], pred[i].shape[3], pred[i].shape[4])
            if not self.args_general.is_condition_control: 
                print('J_total=J_target+w*J_energy=', J_target.mean(), '+', self.args_general.w_energy, '*', J_energy.mean(), '=', J_total.mean())
                print('mae_smoke=', mae_smoke.mean())
            print('mse=', mse.mean(), 'mse_wo_smoke=', mse_wo_smoke.mean())
            print('normalized_l2 (all, density, v1, v2) =', n_l2.mean(), n_l2_density.mean(), n_l2_v1.mean(), n_l2_v2.mean())
                        
            J_totals.append(J_total.mean())
            J_targets.append(J_target.mean())
            J_energys.append(J_energy.mean())
            if method == 'DDPM' and self.args_general.is_super_model:
                mses.append(mse_wo_smoke.mean())
            else:
                mses.append(mse.mean())
            n_l2s.append(n_l2.mean())
        return np.array(J_totals), np.array(J_targets), np.array(J_energys), np.array(mses), np.array(n_l2s) 
    

def inference(dataloader, diffusion, design_fn, args, RESCALER):
    model = diffusion 
    model_args = {
        "design_fn": design_fn,
        "design_guidance": args.design_guidance,
    } 

    inferencePPL = InferencePipeline(
        model, 
        model_args,
        RESCALER,
        results_path = args.inference_result_subpath,
        args_general=args
    )
    inferencePPL.run(dataloader)
    

def load_data(args):
    if args.dataset == "Smoke":
        if args.is_wavelet:
            dataset = Smoke_wave(
                dataset_path=args.dataset_path,
                wave_type=args.wave_type,
                pad_mode=args.pad_mode,
                is_super_model=args.is_super_model,
                N_downsample=0,
            )
            _, shape_init, ori_shape_init, _ = dataset[0]
            if not args.is_super_model:
                shape = shape_init
                ori_shape = ori_shape_init
            else:
                shape, ori_shape = [], []
                shape.append(shape_init)
                ori_shape.append(ori_shape_init)
                for i in range(args.upsample):
                    if not args.is_condition_control:
                        shape.append([2*shape[-1][-3]-2, shape[-1][-2], shape[-1][-1]])
                        ori_shape.append([2*ori_shape[-1][-3], ori_shape[-1][-2], ori_shape[-1][-1]])
                    else:
                        shape.append([shape[-1][-3], 2*shape[-1][-2]-2, 2*shape[-1][-1]-2])
                        ori_shape.append([ori_shape[-1][-3], 2*ori_shape[-1][-2], 2*ori_shape[-1][-1]])
        else:
            dataset = Smoke(
                dataset_path=args.dataset_path,
                is_train=True,
            )
            _, shape, ori_shape, _ = dataset[0]
    else:
        assert False
    RESCALER = dataset.RESCALER.unsqueeze(0).to(args.device)

    dataset = Smoke(
        dataset_path=args.dataset_path,
        is_train=False,
        test_mode='control' if not args.is_condition_control else 'simulation', 
        upsample=args.is_super_model
    ) 
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    print("number of batch in test_loader: ", len(test_loader))
    return test_loader, shape, ori_shape, RESCALER

def main(args):
    dataloader, shape, ori_shape, RESCALER = load_data(args)
    diffusion, design_fn = load_model(args, shape, ori_shape, RESCALER, args.w_energy, w_init=args.w_init) 
    inference(dataloader, diffusion, design_fn, args, RESCALER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference 2d inverse design model')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--dataset', default='Smoke', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--dataset_path', default="path", type=str,
                        help='path to dataset')
    parser.add_argument('--gpu', type=int, default=3, help='gpu id used in training')
    parser.add_argument('--w_energy_list', nargs='+', default=[0], type=float,
                        help='guidance intensity of initial condition')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--upsample', default=0, type=int,
                        help='number of times of upsampling with super resolution model, n *= 2**upsample')
    parser.add_argument('--is_condition_control', default=False, type=eval,
                        help='If condition on control')
    parser.add_argument('--is_condition_pad', default=True, type=eval,
                        help='If condition on padded state')
    parser.add_argument('--is_condition_reward', default=False, type=eval,
                        help='If condition on padded state')
    parser.add_argument('--batch_size', default=50, type=int,
                        help='size of batch of input to use')
    parser.add_argument('--inference_result_path', default="gifs", type=str,
                        help='path to save inference result')
  
    parser.add_argument('--inference_method', default="BPPO", type=str,
                        help='the inference method: DDPM | SAC | PID | SL')

    # DDPM
    parser.add_argument('--diffusion_model_path', default="path", type=str,
                        help='directory of trained diffusion model (Unet)')
    parser.add_argument('--diffusion_checkpoint', default=50, type=int,
                        help='index of checkpoint of trained diffusion model (Unet)')
    parser.add_argument('--super_diffusion_model_path', default="path", type=str,
                        help='directory of trained super diffusion model (Unet)')
    parser.add_argument('--super_diffusion_checkpoint', default=75, type=int,
                        help='index of checkpoint of trained super diffusion model (Unet)')
    parser.add_argument('--using_ddim', default=False, type=eval,
                        help='If using DDIM')
    parser.add_argument('--ddim_eta', default=0., type=float, help='eta in DDIM')
    parser.add_argument('--ddim_sampling_steps', default=100, type=int, 
                        help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')
    parser.add_argument('--design_guidance', default='standard-alpha', type=str,
                        help='design_guidance')
    parser.add_argument('--standard_fixed_ratio_list', nargs='+', default=[0], type=float,
                        help='standard_fixed_ratio for standard sampling')
    parser.add_argument('--coeff_ratio_list', nargs='+', default=[0], type=float,
                        help='coeff_ratio for standard-alpha sampling')
    parser.add_argument('--w_init_list', nargs='+', default=[0], type=float,
                        help='guidance intensity of initial condition')
    parser.add_argument('--alpha', type=float, default=1, metavar='G',
                        help='Temperature parameter \alpha determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    # BPPO
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
    parser.add_argument("--channel", default=5, type=int)
    parser.add_argument('--bppo_model_path', default='path', type=str,
                        help='directory of trained bppo model')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    standard_fixed_ratio_list = args.standard_fixed_ratio_list
    coeff_ratio_list = args.coeff_ratio_list
    w_init_list = args.w_init_list
    w_energy_list = args.w_energy_list

    for w_init in w_init_list:
        for w_energy in w_energy_list:
            for standard_fixed_ratio in standard_fixed_ratio_list:
                for coeff_ratio in coeff_ratio_list:
                    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    args.standard_fixed_ratio = standard_fixed_ratio
                    args.coeff_ratio = coeff_ratio
                    args.w_init = w_init
                    args.w_energy = w_energy
                    args.inference_result_subpath = os.path.join(
                        args.inference_result_path,
                        current_time + "_standard_fixed_ratio_{}".format(args.standard_fixed_ratio) + "_coeff_ratio_{}".format(coeff_ratio)\
                         + "_w_init_{}".format(w_init) + "_w_energy_{}".format(w_energy),
                        args.inference_result_path, 
                    )
                    print("args: ", args)
                    main(args)
    

    