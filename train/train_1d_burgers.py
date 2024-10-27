import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import yaml
import argparse
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from dataset.data_1d import Burgers1D
from diffusion.diffusion_1d_burgers import GaussianDiffusion, Trainer
from model.burgers_1d.unet import Unet2D

none_or_str = lambda x: None if x == 'None' else x


parser = argparse.ArgumentParser(description='Train EBM model')
parser.add_argument('--exp_id', default='gen-control', type=str,
                    help='experiment folder id')
parser.add_argument('--date_time', default=datetime.today().strftime('%Y-%m-%d'), type=str,
                    help='date for the experiment folder')
parser.add_argument('--dataset', default='free_u_f_1e5', type=str,
                    help='dataset name')
parser.add_argument('--train_num_steps', default=100000, type=int,
                    help='train_num_steps')
parser.add_argument('--checkpoint_interval', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')



parser.add_argument('--is_condition_u0', default=False, type=eval,
                    help='If learning p(u_[1, T] | u0)')
parser.add_argument('--is_condition_uT', default=False, type=eval,
                    help='If learning p(u_[0, T-1] | uT)')
parser.add_argument('--is_condition_u0_zero_pred_noise', default=True, type=eval,
                    help='If enforcing the pred_noise to be zero for the conditioned data\
                         when learning p(u_[1, T-1] | u0). if false, mimic bad behavior of exp 0 and 1')
parser.add_argument('--is_condition_uT_zero_pred_noise', default=True, type=eval,
                    help='If enforcing the pred_noise to be zero for the conditioned data\
                         when learning p(u_[1, T-1] | u0). if false, mimic bad behavior of exp 0 and 1')
parser.add_argument('--condition_on_residual', default=None, type=str, 
                    help='option: None, residual_gradient')
parser.add_argument('--residual_on_u0', default=False, type=eval, 
                    help='when using conditioning on residual, whether feeding u0 or ut into Unet')
# exp setting
parser.add_argument('--partially_observed', default=None, type=none_or_str, 
                    help='If None, fully observed, otherwise, partially observed during training\
                        Note that the force is always fuly observed. Possible choices:\
                        front_rear_quarter. \
                        In the training part, partially_observed sets the training trajectories to zero at the unobserved locations')
parser.add_argument('--train_on_partially_observed', default=None, type=none_or_str, 
                    help='Whether to train the model to generate zero states at the unobserved locations. if None, enforce zero.')

# sampling setting: does not affect
parser.add_argument('--set_unobserved_to_zero_during_sampling', default=False, type=eval, 
                    help='Set central 1/2 to zero in each p sample loop.')
parser.add_argument('--recurrence', default=False, type=eval, help='whether to use recurrence in Universal Guidance for Diffusion Models')
parser.add_argument('--recurrence_k', default=1, type=int, help='how many iterations of recurrence. k in Algo 1 in Universal Guidance for Diffusion Models')
parser.add_argument('--using_ddim', default=False, type=eval,
                    help='If using DDIM')
parser.add_argument('--ddim_eta', default=0., type=float, help='eta in DDIM')
parser.add_argument('--ddim_sampling_steps', default=1000, type=int, help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')

# unet hyperparam
parser.add_argument('--dim', default=64, type=int,
                    help='first layer feature dim num in Unet')
parser.add_argument('--resnet_block_groups', default=1, type=int,
                    help='group num in GroupNorm')
parser.add_argument('--dim_muls', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')

# 2 ddpm: learn p(w, u) and p(w) -> use p(u | w) during inference
parser.add_argument('--is_model_w', default=False, type=eval, help='If training the p(w) model, else train the p(u, w) model')
parser.add_argument('--eval_two_models', default=False, type=eval, help='Set to False in this training file')
parser.add_argument('--expand_condition', default=False, type=eval, help='Expand conditioning information of u0 or uT in separate channels')
parser.add_argument('--prior_beta', default=1, type=eval, help='strength of the prior (1 is p(u,w); 0 is p(u|w))')


# In[ ]:


def get_dataset(dataset='free_u_f_1e5', partially_observed=None, nt_total=11):
    return Burgers1D(
        dataset="burgers",
        input_steps=1,
        output_steps=10,
        time_interval=1,
        is_y_diff=False,
        split="train",
        transform=None,
        pre_transform=None,
        verbose=False,
        root_path=f'data/{dataset}' , # dataset of f varying in both space and time
        device='cuda',
        rescaler=10., 
        stack_u_and_f=True, 
        pad_for_2d_conv=True, 
        partially_observed_fill_zero_unobserved = partially_observed, 
        nt_total=nt_total, 
    )



def get_2d_ddpm(args):
    sim_time_stamps, sim_space_grids = 16, 128 # sample * 2 * timegrid (padded to 16) * spacegrid

    # decide channel number.
    # 1 ddpm: u and f in Burger's equation. 
    if args.condition_on_residual is not None and args.condition_on_residual == 'residual_gradient':
        # If conditioned on residual, input dim becomes 4 (nabla_{u,w} residual)
        channels = 4
        if args.is_model_w:
            raise NotImplementedError
    elif args.is_model_w:
        if args.expand_condition:
            channels = 1
            channels += 1 if args.condition_u0 else 0
            channels += 1 if args.condition_uT else 0
        else:
            channels = 2
    else:
        channels = 2

    # make model
    if not args.eval_two_models:
        u_net = Unet2D(
            dim=args.dim, 
            init_dim = None,
            out_dim = 2,
            dim_mults=args.dim_muls,
            channels = channels, 
            self_condition = False,
            resnet_block_groups = args.resnet_block_groups,
            learned_variance = False,
            learned_sinusoidal_cond = False,
            random_fourier_features = False,
            learned_sinusoidal_dim = 16,
            sinusoidal_pos_emb_theta = 10000,
            attn_dim_head = 32,
            attn_heads = 4, 
            condition_on_residual = args.condition_on_residual, 
        )
    ddpm = GaussianDiffusion(
        u_net if not args.eval_two_models else (args.unet_uw, args.unet_w), 
        seq_length=(sim_time_stamps, sim_space_grids), 
        auto_normalize=False, 
        use_conv2d=True, 
        temporal=True, # using temporal conv (2d conv or 1d conv after 1d conv on space)
        is_condition_u0=args.is_condition_u0, 
        is_condition_uT=args.is_condition_uT, 
        is_condition_u0_zero_pred_noise=args.is_condition_u0_zero_pred_noise, 
        is_condition_uT_zero_pred_noise=args.is_condition_uT_zero_pred_noise, 
        train_on_partially_observed=args.train_on_partially_observed, 
        set_unobserved_to_zero_during_sampling=args.set_unobserved_to_zero_during_sampling, 
        conditioned_on_residual = args.condition_on_residual, 
        residual_on_u0 = args.residual_on_u0, 
        recurrence=args.recurrence, 
        recurrence_k=args.recurrence_k, 
        # two ddpm related
        is_model_w=args.is_model_w, 
        eval_two_models=args.eval_two_models, 
        expand_condition=args.expand_condition, 
        prior_beta=args.prior_beta, 
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, 
        ddim_sampling_eta=args.ddim_eta
    )
    return ddpm


def run_2d_Unet(dataset, args):
    """
    Train the 1D Burgers model. 2D Unet here means the Unet takes 2D input (time and space)
    But the Burgers equation is (spatially) 1D.
    """
    ddpm = get_2d_ddpm(args)
    exp_dirname = f"{args.exp_id}/"
    if args.is_model_w:
        results_folder = './trained_models/burgers_w/' + exp_dirname
    else:
        results_folder = './trained_models/burgers/' + exp_dirname
    trainer = Trainer(
        ddpm, 
        dataset, 
        results_folder=results_folder, 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    # trainer.load(1) # load pervious file
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.dataset in ["free_u_f_1e5", 'free_u_f_1e5_front_rear_quarter']:
        dataset = get_dataset(args.dataset, args.partially_observed)
    
    print(f'data shape: {dataset.get(0).shape}')
    print(f'Rescaling data by dividing {dataset.rescaler}')

    # set random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    run_2d_Unet(dataset, args)
