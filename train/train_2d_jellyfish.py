import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from diffusion.diffusion_2d_jellyfish import GaussianDiffusion, Trainer
from model.video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D
import pdb
from accelerate import Accelerator
from filepath import JELLYFISH_DATA_PATH, JELLYFISH_RESULTS_PATH
print("JELLYFISH_DATA_PATH: ", JELLYFISH_DATA_PATH)

import argparse

parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--dataset', default='jellyfish', type=str,
                    help='dataset to evaluate')
parser.add_argument('--dataset_path', default=JELLYFISH_DATA_PATH, type=str,
                    help='path to dataset')
parser.add_argument('--batch_size', default=4, type=int,
                    help='size of batch of input to use')
parser.add_argument('--cond_steps', default=1, type=int,
                    help='number of cond_steps')
parser.add_argument('--frames', default=20, type=int,
                    help='number of frames to diffuse')
parser.add_argument('--traj_len', default=40, type=int,
                    help='number of frames in a whole trajectory')
parser.add_argument('--ts', default=1, type=int,
                    help='timeskip between frames')
parser.add_argument('--is_testdata', default=False, type=bool,
                    help='whether run mini example data, if True, yes; otherwise, run full data')
parser.add_argument('--results_path', default=os.path.join(JELLYFISH_RESULTS_PATH, 'checkpoints'), type=str,
                    help='folder to save training checkpoints')
parser.add_argument('--log_path', default=os.path.join(JELLYFISH_RESULTS_PATH, 'logs'), type=str,
                    help='folder to save training logs')
parser.add_argument('--only_vis_pressure', action='store_true', help="whether only observe pressure")
parser.add_argument('--model_type', default="states", type=str,
                    help='choices: states | thetas')


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    inp_dim = 5 if FLAGS.only_vis_pressure else 7 # use state of velocity and pressure (3), and boundary mask and offset (3), and theta (1)
    fp16 = False
    accelerator = Accelerator(
        split_batches = True,
        mixed_precision = 'fp16' if fp16 else 'no'
    )
    device = accelerator.device
    if FLAGS.model_type == "states":
        model = Unet3D_with_Conv3D(
            dim = 64,
            out_dim = 1 if FLAGS.only_vis_pressure else 3, # velocity, pressure
            dim_mults = (1, 2, 4),
            channels=5 if FLAGS.only_vis_pressure else 7 # use state of velocity and pressure (3), and boundary mask and offset as condition(3), and theta as condition (1)
        )
        results_path = os.path.join(FLAGS.results_path, 'joint_full' if not FLAGS.only_vis_pressure else 'joint_partial')
    elif FLAGS.model_type == "thetas":
        model = Unet3D_with_Conv3D(
            dim = 64,
            out_dim = 1, # theta
            dim_mults = (1, 2, 4),
            channels=5 if FLAGS.only_vis_pressure else 7 # use state of velocity and pressure as condition (3), and boundary mask and offset (3), and theta (1)
        )
        results_path = os.path.join(FLAGS.results_path, 'w_full' if not FLAGS.only_vis_pressure else 'w_partial')
    print("number of parameters in model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Saved at: ", results_path)
    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        frames=FLAGS.frames,
        cond_steps=FLAGS.cond_steps,
        timesteps = 1000,           # number of diffusion steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        objective = "pred_noise",
        device =device
    )

    trainer = Trainer(
        diffusion,
        FLAGS.dataset,
        FLAGS.dataset_path,
        FLAGS.frames,
        FLAGS.traj_len,
        FLAGS.ts,
        FLAGS.log_path,
        train_batch_size = FLAGS.batch_size,
        train_lr = 1e-3,
        train_num_steps = 400000,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 4000,
        results_path = results_path,
        amp = False,                       # turn on mixed precision
        calculate_fid = False,              # whether to calculate fid during training
        is_testdata = FLAGS.is_testdata,
        only_vis_pressure = FLAGS.only_vis_pressure,
        model_type = FLAGS.model_type
    )
    
    trainer.train()