import argparse
from collections import OrderedDict
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import datetime
import matplotlib.pylab as plt
import numpy as np
import time
import torch
from torch.autograd import grad
from accelerate import Accelerator
from dataset.data_2d import Smoke
from diffusion.diffusion_2d_smoke import GaussianDiffusion, Trainer
from dataset.apps.evaluate_solver import *
from model.video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D
from matplotlib.backends.backend_pdf import PdfPages
import multiprocess as mp
from torch.optim import SGD
from filepath import SMOKE_DATA_PATH, SMOKE_RESULTS_PATH


def clip_loss(tensor, min_value, max_value):
    clipped_tensor = torch.clamp(tensor, min_value, max_value)
    
    loss = (tensor - clipped_tensor).abs().mean()
    
    return loss

def guidance_fn(x, args, RESCALER, w_energy=0, w_init=0, low=None, init=None, init_u=None):
    '''
    low, init: rescaled
    init_u: not rescaled
    '''
    x = x * RESCALER

    state = x # [B, 32, 6, 64, 64]
    guidance_success = state[:,-1,-1].mean((-1,-2)).sum()
    # guidance_energy = state[:,3:5].square().mean((1,2,3,4)).sum()
    guidance_energy = state[:,:,3:5].square().mean((1,2,3,4)).sum()
    guidance = -guidance_success + w_energy * guidance_energy
    grad_x = grad(guidance, x, grad_outputs=torch.ones_like(guidance))[0]

    return grad_x

def load_ddpm_model(args, RESCALER):
    # load model_joint
    model_joint = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=6
    )
    # print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model_joint.parameters() if p.requires_grad))
    model_joint.to(args.device)
    diffusion_joint = GaussianDiffusion(
        model_joint,
        image_size = args.image_size,
        frames = 32,
        timesteps = 1000,            # number of steps
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        loss_type = 'l2',            # L1 or L2
        objective = 'pred_noise',
        standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
        coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
        eval_2ddpm = False
    )
    diffusion_joint.eval()
    # load trainer
    trainer = Trainer(
        diffusion_joint,
        dataset = args.dataset,
        dataset_path = args.dataset_path,
        results_path = args.diffusion_model_joint_path, 
        amp = False,                       # turn on mixed precision
    )
    trainer.load(args.diffusion_joint_checkpoint) 
    
    # load model_w
    model_w = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=2
    )
    # print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model_w.parameters() if p.requires_grad))
    model_w.to(args.device)
    diffusion_w = GaussianDiffusion(
        model_w,
        image_size = args.image_size,
        frames = 32,
        timesteps = 1000,            # number of steps
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        loss_type = 'l2',            # L1 or L2
        objective = 'pred_noise',
        standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
        coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
        eval_2ddpm = False
    )
    diffusion_w.eval()
    # load trainer
    trainer = Trainer(
        diffusion_w,
        dataset = args.dataset,
        dataset_path = args.dataset_path,
        results_path = args.diffusion_model_w_path, 
        amp = False,                       # turn on mixed precision
    )
    trainer.load(args.diffusion_w_checkpoint)
    
    diffusion = GaussianDiffusion(
        [diffusion_joint.model, diffusion_w.model],
        image_size = args.image_size,
        frames = 32,
        timesteps = 1000,            # number of steps
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        loss_type = 'l2',            # L1 or L2
        objective = 'pred_noise',
        standard_fixed_ratio = args.standard_fixed_ratio, # used in standard sampling
        coeff_ratio = args.coeff_ratio, # used in standard-alpha sampling
        eval_2ddpm = True,
        w_prob_exp = args.w_prob_exp,
        device = args.device
    )
    
    return diffusion, trainer.device

def load_model(args, RESCALER, w_energy=0, w_init=0):
    RESCALER = RESCALER
    # load main model of each inference method
    if args.inference_method == 'DDPM':
        diffusion, device = load_ddpm_model(args, RESCALER)
        RESCALER = RESCALER.to(device)
    
    # define design function
    def design_fn(x, low=None, init=None, init_u=None):
        grad_x = guidance_fn(x, args, RESCALER, w_energy=w_energy, w_init=w_init, low=low, init=init, init_u=init_u)
    
        return grad_x

    return [diffusion], design_fn
     

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
        self.image_size = self.args_general.image_size
        self.device = self.args_general.device
        self.upsample = args_general.upsample
        self.RESCALER = RESCALER
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    # def save(self, sim_id, pred, results_path):
    #     pred_states, pred_thetas = pred
    #     for index in range(sim_id.shape[0]):
    #         id = sim_id[index].cpu().item()
    #         thetas_filepath = os.path.join(results_path, "thetas", "{}.npy".format(id))
    #         states_filepath = os.path.join(results_path, "states", "{}.npy".format(id))
    #         with open(thetas_filepath, 'wb') as file:
    #             np.save(file, pred_thetas[index].cpu().numpy())
    #         with open(states_filepath, 'wb') as file:
    #             np.save(file, pred_states[index].cpu().numpy())  
    #         print("id {} saved at {}: ".format(id, results_path))

    def run_model(self, state):
        '''
        state: not rescaled
        '''
        state_ori = state.to(self.args_general.device) # [B, 256, 6, 64, 64]
        state = state_ori[:, ::8] # base resolution # [B, 32, 6, 64, 64]
      
        output = self.model[0].sample(
            batch_size = state.shape[0],
            design_fn=self.args["design_fn"],
            design_guidance=self.args["design_guidance"],
            low=None, 
            init=state[:,0,0]/self.RESCALER[:,0,0], # 
            init_u=state[:,0,0], # only used in guidance if is_wavelet
            control=state[:,:,3:5]/self.RESCALER[:,:,3:5]
            )
        output = output * self.RESCALER
        output[:,:,-1] = output[:,:,-1].mean((-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,64,64)
        return output

    import matplotlib.pyplot as plt
    def plot_time(self, coef, shape, ori_shape):
        for idx in range(5):
            fig, axes = plt.subplots(nrows=1, ncols=8, dpi=300)
            fig.set_size_inches(42, 5)
            ax = axes[0]
            im = ax.imshow(coef[1]['aad'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')  
            ax.set_title(f'diffused W_f, layer=aad')

            ax = axes[1]
            im = ax.imshow(coef[1]['ada'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_f, layer=ada')

            ax = axes[2]
            im = ax.imshow(coef[1]['daa'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_f, layer=daa')

            ax = axes[3]
            im = ax.imshow(coef[1]['dad'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_f, layer=dad')

            ax = axes[4]
            im = ax.imshow(coef[1]['dda'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_f, layer=dda')

            ax = axes[5]
            im = ax.imshow(coef[1]['add'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_f, layer=add')

            ax = axes[6]
            im = ax.imshow(coef[1]['ddd'].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_f, layer=ddd')

            ax = axes[7]
            im = ax.imshow(coef[0].reshape(-1,5,shape[0],shape[1],shape[2])[idx, 3, :, -5].abs().cpu().numpy(), cmap='viridis')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Value')
            ax.set_title(f'diffused W_u, layer=coarse')

            save_path = 'results/figs/plot_layer_{}.jpg'.format(idx)
            plt.savefig(save_path)
            plt.close()
        print(save_path)
        assert 1==2


    def run(self, dataloader):
        preds = []
        J_totals, J_targets, J_energys, mses, n_l2s = {}, {}, {}, {}, {}
        for i in range(self.upsample+1): # super resolution
            J_totals[i], J_targets[i], J_energys[i], mses[i], n_l2s[i] = [], [], [], [], []
        for i, data in enumerate(dataloader):
            print(f"Batch No.{i}")
            state, sim_id = data # shape and ori_shape are not neeeded for baselines
            if self.args_general.inference_method == "DDPM":
                pred = self.run_model(state)
            print("pred shape: ", pred.shape)
            preds.append(pred)
            J_total, J_target, J_energy, mse, n_l2 = self.multi_evaluate(pred, state, plot=False, method=self.args_general.inference_method)
            J_totals[0].append(J_total)
            J_targets[0].append(J_target)
            J_energys[0].append(J_energy)
            mses[0].append(mse)
            n_l2s[0].append(n_l2)
            
            torch.cuda.empty_cache()

        print("Final results!")
        # save
        for i in range(self.upsample+1):
            print(f"Number of upsampling times: {i}")
            print(f"J_total: {np.stack(J_totals[i]).mean(0)},\nJ_target: {np.stack(J_targets[i]).mean(0)},\nJ_energy: {np.stack(J_energys[i]).mean(0)},\nmse: {np.stack(mses[i]).mean(0)},\nn_l2: {np.stack(n_l2s[i]).mean(0)}")
        save_file = 'results.txt'
        if self.args_general.inference_method == "SL":
            results_path=self.args_general.inference_result_path
        else:
            results_path=self.results_path
        with open(os.path.join(results_path, save_file), 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+'\n')
            f.write(str(self.args_general)+'\n')
            for i in range(self.upsample+1):
                f.write(f"Number of upsampling times: {i}\n")
                f.write(f"J_total: {np.stack(J_totals[i]).mean(0)},\nJ_target: {np.stack(J_targets[i]).mean(0)},\nJ_energy: {np.stack(J_energys[i]).mean(0)},\nmse: {np.stack(mses[i]).mean(0)},\nn_l2: {np.stack(n_l2s[i]).mean(0)}\n")
            f.write("-----------------------------------------------------------------------------------------\n")
        # self.save(sim_id, pred, self.results_path) 
    
    def per_evaluate(self, sim, eval_no, pred, data, output_queue):
        '''
        eval_no: No of multi-process
        pred: torch.Tensor, [nt, 6, nx, nx]
        '''
        # print(f'Evaluate No.{eval_no}')
        init_velocity = init_velocity_() # 1, 128, 128, 2
        init_density = data[0,0,:,:] # nx, nx
        c1 = pred[:,3,:,:] # nt, nx, nx
        c2 = pred[:,4,:,:] # nt, nx, nx
        per_solver_out = solver(sim, init_velocity, init_density, c1, c2, per_timelength=256)
        # print(f'Evaluate No.{eval_no} down!')
        try:
            output_queue.put({eval_no:per_solver_out})
            # print(f"Queue Put down {eval_no}")
        except Exception as e:
            print(f"Error in process {eval_no}: {e}")

    def multi_evaluate(self, pred, data, plot=False, method='DDPM'):
        '''
        pred: torch.Tensor, [B, 32, 6, 64, 64] 
        data: torch.Tensor, control: [B, 256, 6, 64, 64], simulation: [B, 32, 6, 128, 128]
        '''
        pred[:, 0, 0] = data[:, 0, 0, ::int(data.shape[-1]/pred.shape[-1]), ::int(data.shape[-1]/pred.shape[-1])] # initial condition
        # control 
        print("Start solving...")
        start = time.time()
        pool_num = pred.shape[0]
        solver_out = np.zeros((pred.shape[0],256,6,128,128), dtype=float)
        pred_ = pred.detach().cpu().numpy().copy()
        data_ = data.detach().cpu().numpy().copy()
        pred_[:,:,3:5,8:56,8:56] = 0 # indirect control
        
        # save pred_ as npy
        print("pred_.shape: ", pred_.shape)
        np.save('plot_pred.npy', pred_)
        print("end save")

        
        sim = init_sim_128()
        output_queue = mp.Queue()

        processes = []
        args_list = [(sim, i, pred_[i,:,:,:,:].copy(), data_[i,:,:,:,:].copy(),output_queue) for i in range(pool_num)]

        for args in args_list:
            process = mp.Process(target=self.per_evaluate, args=args)
            processes.append(process)
            process.start()

        # print(f"Total processes started: {len(processes)}")

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

        # print("Process Join Down!")
        for i in range(len(multi_results_sorted)):
            solver_out[i,:,0,:,:] = multi_results_sorted[f'{i}'][0] # density
            # solver_out[i,:,1,:,:] = multi_results_sorted[f'{i}'][1] # zero_density
            solver_out[i,:,1,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,0] # vel_x
            solver_out[i,:,2,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,1] # vel_x
            solver_out[i,:,3,:,:] = multi_results_sorted[f'{i}'][3] # control_x
            solver_out[i,:,4,:,:] = multi_results_sorted[f'{i}'][4] # control_y
            solver_out[i,:,5,:,:] = multi_results_sorted[f'{i}'][5] # smoke_portion

        if plot:
        # if True:
            print("Start Generating GIFs...")
            """
            Generate GIF
            """
            for i in range(10,50):
                # gif_control(solver_out[i,:,3,:,:], solver_out[i,:,4,:,:],control_bool=True,name=f'{i}')
                # # gif_density(solver_out[i,:,1,:,:],zero=True)
                # gif_control(solver_out[i,:,1,:,:], solver_out[i,:,4,:,:],control_bool=False,name=f'{i}')
                gif_density(solver_out[i,:,0,:,:],zero=False,name=f'WDNO{i}')
        
        data_super = torch.tensor(solver_out, device=pred.device)[:, :, :, ::2, ::2] # no space super resolution
        data_current = data_super[:, ::int(data_super.shape[1]/pred.shape[1])]
        data_base = data_super[:, ::8]
        end = time.time()
        print(f"Time cost: {end-start}")
        pred_current = pred
            
        data = [data_current]
        pred = [pred_current]
        # embed()
        J_totals, J_targets, J_energys, mses, n_l2s = [], [], [], [], []
        for i in range(len(pred)):
            mask = torch.ones_like(pred[i], device = pred[i].device)
            mask[:, 0] = False
            pred[i] = pred[i] * mask 
            data[i] = data[i] * mask
            # embed() # (((pred[i] - data[i])[:,:,3].square().sum((1, 2, 3))).sqrt()/data[i][:,:,3].square().sum((1, 2, 3)).sqrt()).detach().cpu().numpy()
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
            print('J_total=J_target+w*J_energy=', J_target.mean(), '+', self.args_general.w_energy, '*', J_energy.mean(), '=', J_total.mean())
            print('J_total=', J_total)
            print('mae_smoke=', mae_smoke.mean())
            print('mse=', mse.mean(), 'mse_wo_smoke=', mse_wo_smoke.mean())
            print('normalized_l2 (all, density, v1, v2) =', n_l2.mean(), n_l2_density.mean(), n_l2_v1.mean(), n_l2_v2.mean())
            J_totals.append(J_total.mean())
            J_targets.append(J_target.mean())
            J_energys.append(J_energy.mean())
            mses.append(mse.mean())
            n_l2s.append(n_l2.mean())
        return np.array(J_totals), np.array(J_targets), np.array(J_energys), np.array(mses), np.array(n_l2s) # [4,]
    

def inference(dataloader, diffusion, design_fn, args, RESCALER):
    model = diffusion # may vary according to different control methods
    model_args = {
        "design_fn": design_fn,
        "design_guidance": args.design_guidance,
    } # may vary according to different control methods

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
        dataset = Smoke(
            dataset_path=args.dataset_path,
            is_train=True,
        )
        _, _ = dataset[0]
    else:
        assert False
    RESCALER = dataset.RESCALER.unsqueeze(0).to(args.device)

    dataset = Smoke(
        dataset_path=args.dataset_path,
        is_train=False,
    ) # the super resolution data, super_nt=8*base_nt, super_nx=2*base_nx, not rescaled
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = False, pin_memory = True, num_workers = 32)
    print("number of batch in test_loader: ", len(test_loader))

    return test_loader, RESCALER

def main(args):
    dataloader, RESCALER = load_data(args)
    diffusion, design_fn = load_model(args, RESCALER, args.w_energy, w_init=args.w_init) # may vary according to different control methods
    inference(dataloader, diffusion, design_fn, args, RESCALER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference 2d inverse design model')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--dataset', default='Smoke', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--dataset_path', default=SMOKE_DATA_PATH, type=str,
                        help='path to dataset')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
    parser.add_argument('--w_energy', default=0, type=float,
                        help='guidance intensity of initial condition')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--upsample', default=0, type=int,
                        help='number of times of upsampling with super resolution model, n *= 2**upsample')

    parser.add_argument('--is_condition_pad', default=True, type=eval,
                        help='If condition on padded state')
    parser.add_argument('--is_condition_reward', default=False, type=eval,
                        help='If condition on padded state')
    parser.add_argument('--batch_size', default=50, type=int,
                        help='size of batch of input to use')
    parser.add_argument('--inference_result_path', default=os.path.join(SMOKE_RESULTS_PATH, 'inference_results'), type=str,
                        help='path to save inference result')
  
    parser.add_argument('--inference_method', default="DDPM", type=str,
                        help='the inference method: DDPM | SAC | PID | SL')

    # DDPM
    parser.add_argument('--diffusion_model_joint_path', default=os.path.join(SMOKE_RESULTS_PATH, 'checkpoints/joint_models'),
                        type=str,
                        help='directory of trained diffusion joint model (Unet)')
    parser.add_argument('--diffusion_joint_checkpoint', default=50, type=int,
                        help='index of checkpoint of trained diffusion joint model (Unet)')
    parser.add_argument('--diffusion_model_w_path', default=os.path.join(SMOKE_RESULTS_PATH, 'checkpoints/w_models'), 
                        type=str,
                        help='directory of trained diffusion w model (Unet)')
    parser.add_argument('--diffusion_w_checkpoint', default=17, type=int,
                        help='index of checkpoint of trained diffusion w model (Unet)')
    parser.add_argument('--using_ddim', default=True, type=eval,
                        help='If using DDIM')
    parser.add_argument('--ddim_eta', default=1., type=float, help='eta in DDIM')
    parser.add_argument('--w_prob_exp', default=0.97, type=float, 
                        help='gamma of prior reweighting in the paper, should be within (0,1], if 1.0 means no reweighting, i.e., DiffPhyCon-lite')
    parser.add_argument('--ddim_sampling_steps', default=100, type=int, 
                        help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')
    parser.add_argument('--design_guidance', default='standard', type=str,
                        help='design_guidance')
    parser.add_argument('--standard_fixed_ratio', default=100000, type=float,
                        help='standard_fixed_ratio for standard sampling')
    parser.add_argument('--coeff_ratio', default=0, type=float,
                        help='standard_fixed_ratio for standard-alpha sampling')
    parser.add_argument('--w_init', default=0, type=float,
                        help='guidance intensity of initial condition')


    # get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    args.inference_result_subpath = os.path.join(
        args.inference_result_path,
        current_time
    )
    print("args: ", args)
    main(args)
