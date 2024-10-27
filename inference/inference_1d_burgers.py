import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import copy
import torch
import argparse
import numpy as np
from diffusion.diffusion_1d_burgers import Trainer, cosine_beta_J_schedule, get_nablaJ, plain_cosine_schedule, sigmoid_schedule, sigmoid_schedule_flip
from dataset.apps.generate_burgers import burgers_numeric_solve_free
from train.train_1d_burgers import get_2d_ddpm
from utils import mse_deviation, mse_dist_reg, ddpm_guidance_loss, load_burgers_dataset, burgers_metric, get_target

none_or_str = lambda x: None if x == 'None' else x
RESCALER = 10

parser = argparse.ArgumentParser(description='Eval EBM model')
parser.add_argument('--exp_id', type=str,
                    help='trained model id')
parser.add_argument('--model_str_in_key', default='', type=str,
                    help='description that appears in the result dict key')
parser.add_argument('--save_file', default='burgers_results/result_zerowf.yaml', type=str,
                    help='file to save')
parser.add_argument('--dataset', default='free_u_f_1e5', type=str,
                    help='dataset name for evaluation (eval samples drawn from)')
parser.add_argument('--model_str', default='', type=str,
                    help='description (where is this used?)')
parser.add_argument('--n_test_samples', default=50, type=int,
                    help='First n_test_samples samples will be used for evaluation')

# experiment settings
parser.add_argument('--partial_control', default='full', type=none_or_str,
                    help='partial control setting. full or None if fully controlled')
parser.add_argument('--partially_observed', default=None, type=none_or_str, 
                    help='If None, fully observed, otherwise, partially observed during training\
                        Note that the force is always fuly observed. Possible choices:\
                        front_rear_quarter')
parser.add_argument('--train_on_partially_observed', default=None, type=none_or_str, 
                    help='Whether to train the model to generate zero states at the unobserved locations.')
parser.add_argument('--set_unobserved_to_zero_during_sampling', default=False, type=eval, 
                    help='Set central 1/2 to zero in each p sample loop.')

# p(u, w) model training
parser.add_argument('--checkpoint', default=10, type=int,
                    help='which checkpoint model to load')
parser.add_argument('--checkpoint_interval', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')
parser.add_argument('--train_num_steps', default=100000, type=int,
                    help='train_num_steps')

# sampling 
parser.add_argument('--using_ddim', default=False, type=eval,
                    help='If using DDIM')
parser.add_argument('--ddim_eta', default=0., type=float, help='eta in DDIM')
parser.add_argument('--ddim_sampling_steps', default=1000, type=int, help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')
parser.add_argument('--J_scheduler', default=None, type=str,
                    help='which J_scheduler to use. None means no scheduling.')
parser.add_argument('--recurrence', default=False, type=eval, help='whether to use recurrence in Universal Guidance for Diffusion Models')
parser.add_argument('--recurrence_k', default=1, type=int, help='how many iterations of recurrence. k in Algo 1 in Universal Guidance for Diffusion Models')
# usage: python eval.py --wfs 0 1 2 3
parser.add_argument('--wfs', nargs='+', default=[0], type=float,
                    help='guidance intensity of energy')
parser.add_argument('--wus', nargs='+', default=[0], type=float,
                    help='guidance intensity of state deviation')
parser.add_argument('--wreg', default=0, type=float,
                    help='guidance intensity of proximity regularization')
parser.add_argument('--wpinns', nargs='+', default=[0], type=float,
                    help='guidance intensity of state deviation')
parser.add_argument('--pinn_loss_mode', default='mean', type=str,
                    help="Mode of PINN loss evaluation. Choose from ('mean', 'forward', 'backward)")
# residual
parser.add_argument('--condition_on_residual', default=None, type=str, 
                    help='option: None, residual_gradient')
parser.add_argument('--residual_on_u0', default=False, type=eval, 
                    help='when using conditioning on residual, whether feeding u0 or ut into Unet')

# ddpm and unet
parser.add_argument('--is_condition_u0', default=False, type=eval,
                    help='If learning p(u_[1, T-1] | u0)')
parser.add_argument('--is_condition_uT', default=False, type=eval,
                    help='If learning p(u_[0, T-1] | uT)')
parser.add_argument('--is_condition_u0_zero_pred_noise', default=True, type=eval,
                    help='If enforcing the pred_noise to be zero for the conditioned data\
                         when learning p(u_[1, T-1] | u0). if false, mimic bad behavior of exp 0 and 1')
parser.add_argument('--is_condition_uT_zero_pred_noise', default=True, type=eval,
                    help='If enforcing the pred_noise to be zero for the conditioned data\
                         when learning p(u_[1, T-1] | u0). if false, mimic bad behavior of exp 0 and 1')

# unet hyperparam
parser.add_argument('--dim', default=64, type=int,
                    help='first layer feature dim num in Unet')
parser.add_argument('--resnet_block_groups', default=1, type=int,
                    help='')
parser.add_argument('--dim_muls', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')

# two ddpm: learn p(w, u) and p(w) -> use p(u | w) during inference
# NOTE: p(w) model is now default on condition on u0 and uT
parser.add_argument('--is_model_w', default=False, type=eval, help='If training the p(w) model, else train the p(u, w) model')
parser.add_argument('--eval_two_models', default=False, type=eval, help='Set to False in this training file')
parser.add_argument('--expand_condition', default=False, type=eval, help='Expand conditioning information of u0 or uT in separate channels')
parser.add_argument('--prior_beta', default=1, type=float, help='strength of the prior (1 is p(u,w); 0 is p(u|w))')
parser.add_argument('--normalize_beta', default=False, type=eval, help='')
parser.add_argument('--w_scheduler', default=None, type=none_or_str,
                    help='which scheduler to use in front of nabla log p(w). None means no scheduling.')

# arguments below (with double underlined "__model_w") are only used in eval two ddpm: these args will overwrite the above model args when loading the p(w) model. See use_args_w in test_utils.py
parser.add_argument('--exp_id__model_w', type=str,
                    help='trained model id')
# training
parser.add_argument('--checkpoint__model_w', default=10, type=int,
                    help='which checkpoint model to load')
parser.add_argument('--checkpoint_interval__model_w', default=10000, type=int,
                    help='save checkpoint every checkpoint_interval steps')
parser.add_argument('--train_num_steps__model_w', default=100000, type=int,
                    help='train_num_steps')
# unet
parser.add_argument('--dim__model_w', default=64, type=int,
                    help='first layer feature dim num in Unet')
parser.add_argument('--resnet_block_groups__model_w', default=1, type=int,
                    help='')
parser.add_argument('--dim_muls__model_w', nargs='+', default=[1, 2, 4, 8], type=int,
                    help='dimension of channels, multiplied to the base dim\
                        seq_length % (2 ** len(dim_muls)) must be 0')


# loss, guidance and utils

def get_loss_fn_2dconv(
        wf=0, wu=0, wpinn=0, target_i=0, 
        wu_eval=1, wf_eval=0, device=0, 
        dataset='free_u_f_1e5', 
        dist_reg=lambda x: 0, 
        wreg=0, 
        partially_observed=None, 
        pinn_loss_mode='mean', # how to calculate pinn loss
):
    u_target = get_target(
        target_i, 
        device=device, 
        dataset=dataset, 
        partially_observed_fill_zero_unobserved=partially_observed
    )
    def loss_fn_2dconv(x, eval=False):
        # x: of shape (batch, 2, num_time, num_grids), actual scaled
        if eval:
            # report metric of the diffused state
            raise NotImplementedError('Should not have used this branch. Loss reported also using the custom metric function')
            return ddpm_guidance_loss(
                u_target, x[:, 0, :11, :], x[:, 1, :10, :], 
                wu=wu_eval, 
                wf=wf_eval, 
                partially_observed=None
            )
        else:
            # use rescaled value in guidance
            return ddpm_guidance_loss(
                u_target / RESCALER, x[:, 0, :11, :], x[:, 1, :10, :], 
                wu=wu, wf=wf, wpinn=wpinn, 
                dist_reg=dist_reg, 
                pinn_loss_mode=pinn_loss_mode, 
                wreg=wreg, 
                partially_observed=partially_observed, # only calculate guidance on the observed locations
            )
    return loss_fn_2dconv

def get_nablaJ_2dconv(**kwargs):
    return get_nablaJ(get_loss_fn_2dconv(**kwargs))


def use_args_w(args):
    args = copy.deepcopy(args)
    model_w_key = '__model_w'
    for k in args.__dict__.keys():
        if model_w_key in k:
            setattr(args, k[:-len(model_w_key)], getattr(args, k))
    
    return args


def load_2dconv_model_two_ddpm(i, args):
    args = copy.deepcopy(args) # was a bug... should not modify the arg used in the outer scope...
    dataset = load_burgers_dataset()
    # load p(u, w)
    args.is_ddpm_w = False
    args.eval_two_models = False # when loading the separately trained model, should not use eval_two_models
    ddpm_uw = get_2d_ddpm(args)
    trainer = Trainer(
        ddpm_uw, 
        dataset, 
        results_folder=f'./trained_models/burgers/{args.exp_id}/', 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    unet_uw = ddpm_uw.model

    # load p(w)
    args.is_ddpm_w = True
    args.eval_two_models = False
    args = use_args_w(args)
    ddpm_w = get_2d_ddpm(args)
    trainer = Trainer(
        ddpm_w, 
        dataset, 
        results_folder=f'./trained_models/burgers_w/{args.exp_id__model_w}/', 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    unet_w = ddpm_w.model

    args.eval_two_models = True
    args.is_ddpm_w = False
    args.unet_uw = unet_uw
    args.unet_w = unet_w
    ddpm_two_models = get_2d_ddpm(args)
    return ddpm_two_models.cuda()
    
    
    
def load_2dconv_model(i, args, new=True):
    if args.eval_two_models:
        assert not args.is_model_w
        return load_2dconv_model_two_ddpm(i, args)
    elif args.is_model_w:
        return load_2dconv_model_w(args)
    
    dataset = load_burgers_dataset()
    ddpm = get_2d_ddpm(args)

    trainer = Trainer(
        ddpm, 
        dataset, 
        results_folder=f"./trained_models/burgers/{i}/", 
        train_num_steps=args.train_num_steps, 
        save_and_sample_every=args.checkpoint_interval, 
    )
    trainer.load(args.checkpoint if 'checkpoint' in args.__dict__ else 10)
    return ddpm


def load_2dconv_model_w(args):
    # copy args for loading model w
    args = use_args_w(copy.deepcopy(args))
    dataset = load_burgers_dataset()
    ddpm = get_2d_ddpm(args)

    trainer = Trainer(
        ddpm, 
        dataset, 
        results_folder=f"./trained_models/burgers_w/{args.exp_id__model_w}/", 
        train_num_steps=args.train_num_steps__model_w, 
        save_and_sample_every=args.checkpoint__model_w, 
    )
    trainer.load(args.checkpoint__model_w)
    return ddpm

# run exp

def diffuse_2dconv(args, custom_metric, model_i, seed=0, ret_ls=False, **kwargs):
    '''
    data size: (
        batch, 
        2 (stacked from u and f), 
        16 (padded from 10 and 11 respectively), 
        128
    )

    Returns: 
        ddpm_mse: (batch)
        J_diffused: (reported_metric, batch)
        J_actual: (reported_metric, batch)
        energy: (batch)
    '''
    # helper
    u_from_x = lambda x: x[:, 0, :11, :]
    u0_from_x = lambda x: x[:, 0, 0, :]
    f_from_x = lambda x: x[:, 1, :10, :]

    # run 5 different seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(0)
    ddpm = load_2dconv_model(model_i, args)
    
    # sample: actual scaled x and x_gt
    # import time
    # t1 = time.time()
    x = ddpm.sample(**kwargs) * RESCALER
    # t2 = time.time()
    # print(f'Sampling time: {t2 - t1} s')
    # BUG: with this implementation, when partially observed and using zero filling, this x_gt is intricically biased \
    # from the ground truth and thus cannot be used as a sign of how good DDPM learns the physical rules
    x_gt = burgers_numeric_solve_free(u0_from_x(x), f_from_x(x), visc=0.01, T=1.0, dt=1e-4, num_t=10)
    
    ddpm_mse = mse_deviation(u_from_x(x), x_gt, partially_observed=args.partially_observed).cpu()
    J_diffused, _ = custom_metric(f_from_x(x), diffused_u=u_from_x(x), evaluate_u=True)
    J_actual, energy = custom_metric(f_from_x(x))
    # If J contains multiple metrics, move to cpu and convert to numpy one by one
    elems_to_cpu_numpy_if_tuple = lambda x: x.cpu().numpy() if type(x) is not tuple else np.array([xi.cpu().numpy() for xi in x])
    J_diffused = elems_to_cpu_numpy_if_tuple(J_diffused)
    J_actual = elems_to_cpu_numpy_if_tuple(J_actual)
    
    energy = energy.cpu().numpy()
    return ddpm_mse, J_diffused, J_actual, energy


def get_scheduler(scheduler):
    if scheduler is None:
        return None
    # decreasing schedules
    elif scheduler == 'linear':
        raise NotImplementedError
    elif scheduler == 'cosine':
        return cosine_beta_J_schedule
    elif scheduler == 'plain_cosine':
        return plain_cosine_schedule
    elif scheduler == 'sigmoid':
        return sigmoid_schedule
    # increasing step (eta[t=0] is the largest)
    elif scheduler == 'sigmoid_flip':
        return sigmoid_schedule_flip
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")
    

def evaluate(
        model_i, 
        args, 
        # exp settings
        # guidance choices
        wu=0, 
        wf=0, 
        wpinn=0, 
        wf_eval=0, 
        wu_eval=1, 
        # model choice
        conv2d=True, 
):
    n_test_samples = args.n_test_samples
    batch_size = 50
    assert n_test_samples % batch_size  == 0
    rep = n_test_samples // batch_size

    mses = []
    l_gts = []
    l_dfs = []
    energies = []
    for i in range(rep):
        seed = i
        target_idx = list(range(i * batch_size, (i + 1) * batch_size)) # should also work if being an iterable
        if conv2d:
            _, _, J_actual, energy = diffuse_2dconv(
                args, 
                # get_loss_fn_2dconv(wu=wu, wf=wf, target_i=target_idx, dataset=args.dataset, partially_observed=args.partially_observed), 
                custom_metric=lambda f, **kwargs: burgers_metric(
                    get_target(target_idx, dataset=args.dataset), 
                    f, 
                    target='final_u', 
                    partial_control=args.partial_control, 
                    report_all=True, # J_actual will be a 4 tuple
                    partially_observed=args.partially_observed, 
                    **kwargs
                ), 
                model_i=model_i, 
                seed=seed, 
                # more ddpm settings
                nablaJ=get_nablaJ_2dconv(
                    target_i=target_idx, 
                    wu=wu, wf=wf, wpinn=wpinn, 
                    wf_eval=wf_eval, wu_eval=wu_eval, 
                    dist_reg=mse_dist_reg, wreg=args.wreg,
                    dataset=args.dataset, 
                    partially_observed=args.partially_observed, 
                    pinn_loss_mode=args.pinn_loss_mode, 
                ),  
                J_scheduler=get_scheduler(args.J_scheduler), 
                w_scheduler=get_scheduler(args.w_scheduler), 
                # proj_guidance=get_proj_ep_orthogonal_func(norm='F'), 
                clip_denoised=True, 
                guidance_u0=True, 
                batch_size=batch_size, # rep * 5 = first 50 samples
                u_init=get_target(
                    target_idx, 
                    dataset=args.dataset, 
                    partially_observed_fill_zero_unobserved=args.partially_observed
                )[:, 0, :] / RESCALER, # this will not be used when not is_condition_u0
                u_final=get_target(
                    target_idx, 
                    dataset=args.dataset, 
                    partially_observed_fill_zero_unobserved=args.partially_observed
                )[:, 10, :] / RESCALER, # this will not be used when not is_condition_uT
            )
        # dimension zero: repetition
        l_gts.append(J_actual)
        energies.append(energy)

        print('J_actual:', l_gts[0][0].mean())
        print('Energy:', energies[0].mean())
    
    
if __name__ == '__main__':
    args = parser.parse_args()

    model_i, model_str, conv_2d = args.exp_id, args.model_str, True
    
    results = evaluate(
        model_i = model_i, # see trained_model/log.yaml for reference
        args = args,
        conv2d = conv_2d, 
    )