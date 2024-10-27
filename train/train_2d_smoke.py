import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from diffusion.diffusion_2d_smoke import GaussianDiffusion, Trainer
from dataset.data_2d import Smoke
from model.video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D
import torch
from filepath import SMOKE_DATA_PATH, SMOKE_RESULTS_PATH

from accelerate import Accelerator
import argparse


parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--dataset', default='Smoke', type=str,
                    help='dataset to evaluate')
parser.add_argument('--dataset_path', default=SMOKE_DATA_PATH, type=str,
                    help='path to dataset')
parser.add_argument('--batch_size', default=6, type=int,
                    help='size of batch of input to use')
parser.add_argument('--train_num_steps', default=200000, type=int,
                    help='total training steps')
parser.add_argument('--is_w_model', action='store_true',
                    help='whether to train w model')
parser.add_argument('--results_path', default=os.path.join(SMOKE_RESULTS_PATH, 'checkpoints'), type=str,
                    help='folder to save training checkpoints')


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    print(FLAGS)
    
    # get shape, RESCALER
    dataset = Smoke(
        dataset_path=FLAGS.dataset_path,
        is_train=True,
    )
    _, _ = dataset[0]

    RESCALER = dataset.RESCALER.unsqueeze(0)

    channels = 6 if not FLAGS.is_w_model else 2
    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels = channels,
    )
    print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    results_path = os.path.join(FLAGS.results_path, 'w' if FLAGS.is_w_model else 'joint')
    print("Saved at: ", results_path)

    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        frames = 32,
        timesteps = 1000,           # number of diffusion steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l2',            # L1 or L2
        objective = "pred_noise",
    )

    trainer = Trainer(
        diffusion,
        FLAGS.dataset,
        FLAGS.dataset_path,
        train_batch_size = FLAGS.batch_size,
        train_lr = 1e-3, 
        train_num_steps = FLAGS.train_num_steps, # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        save_and_sample_every = 10000,
        results_path = results_path,
        amp = False,                       # turn on mixed precision
        is_w_model=FLAGS.is_w_model
    )

    trainer.train()