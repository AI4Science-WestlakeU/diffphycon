import numpy as np
import torch
import pdb
import sys, os
import matplotlib.pylab as plt
import argparse
import math
import logging
from diffusion_2d import ForceUnet, Unet
from utils import update_static_masks
from data_2d import Jellyfish
from utils import p

def get_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_id))
    else:
        device = torch.device("cpu")
    return device

class SurrogatePipeline(object):
    def __init__(
        self,
        simulator_model,
        force_model,
        boundary_updater_model
    ):
        super().__init__()
        self.simulator_model = simulator_model
        self.force_model = force_model
        self.boundary_updater_model = boundary_updater_model
        self.simulator_model.eval()
        self.force_model.eval()
        self.m = 20 # number of points on each ellipse wing of jellyfish
        self.frames = 20 # number of time steps of each simulation
    
    
    def update_mask_offsets(self, mask_offset_0, theta):
        """ 
        update mask and offset of boundary based on boundary points and theta
        input:
          mask_offset_0: (batch_size, 3, 128, 128), 3 channels: mask, offset_x, offset_y of boundary of the INITIAL time step
          theta: (batch_size), angle of the upper wing of the next time step
        output:
          mask_offsets: (batch_size, 3, 128, 128), 3 channels: mask, offset_x, offset_y of boundary condition on theta angle and the INITIAL boundary
        """
        return self.boundary_updater_model(mask_offset_0, theta)
        
    
    def update_mask_offsets_new(self, mask_offset_t, theta_delta):
        """ 
        update mask and offset of boundary based on boundary points and theta
        input:
          mask_offset_t: (batch_size, 3, 128, 128), 3 channels: boundary mask and offset of time t
          theta_delta: (batch_size), angel difference in arc of from time t to t+1
        output:
          mask_offsets: (batch_size, 3, 128, 128), 3 channels: boundary mask and offset of time t+1
        """
        return self.boundary_updater_model(mask_offset_t, theta_delta)
    

    def run(self, state, mask_offsets, theta_delta=None):
        """
        compute the output force_x of next time step based on state, mask_offsets of boundary and theta of current time step
        input:
            state: (batch_size, 3, 128, 128), 3 channels: (v_x, v_y, pressure) of current time step
            mask_offsets: (batch_size, 3, 128, 128), 3 channels: mask, offset_x, offset_y of boundary of current time step
            theta_delta: (batch_size), angel difference in arc of from time t to t+1. 
                         None: output does not contain states_next; not None: output contains states_next. 
        output:
            states_next: (batch_size, 3, 128, 128), 3 channels: (v_x, v_y, pressure) of next time step
            force_x: (batch_size), horizontal force of current time step
        """
        x = torch.cat((state, mask_offsets), dim=1)
        with torch.no_grad():
            pressure = state[:, -1, :, :]
            input = torch.cat((pressure.unsqueeze(1), mask_offsets), dim=1)
            force = self.force_model(input)
            force_x = force[:, 0]
            if theta_delta != None:
                states_next = self.simulator_model(x, theta_delta)            
                return states_next, force_x 
            else: 
                return force_x

    def run_vis_pressure(self, state, mask_offsets, theta_delta=None):
        """
        compute the output force_x of next time step based on state, mask_offsets of boundary and theta of current time step
        input:
            state: (batch_size, 1, 128, 128), 1 channels: (pressure) of current time step
            mask_offsets: (batch_size, 3, 128, 128), 3 channels: mask, offset_x, offset_y of boundary of current time step
            theta_delta: (batch_size), angel difference in arc of from time t to t+1. 
                         None: output does not contain states_next; not None: output contains states_next. 
        output:
            states_next: (batch_size, 1, 128, 128), 1 channels: (pressure) of next time step
            force_x: (batch_size), horizontal force of current time step
        """
        x = torch.cat((state, mask_offsets), dim=1)
        with torch.no_grad():
            pressure = state[:, 0, :, :]
            input = torch.cat((pressure.unsqueeze(1), mask_offsets), dim=1)
            force = self.force_model(input)
            force_x = force[:, 0]
            if theta_delta != None:
                states_next = self.simulator_model(x, theta_delta)            
                return states_next, force_x 
            else: 
                return force_x
            

def build_ppl_new(args):
    device = get_device(args.gpu)
    force_model = ForceUnet(
        dim = args.image_size,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        channels=4
    )
    boundary_updater_model = Unet(
        dim = args.image_size,
        out_dim = 3,
        dim_mults = (1, 2, 4, 8),
        channels=3
    )
    if args.only_vis_pressure:
        simulator_model = Unet(
            dim = args.image_size,
            out_dim = 1,
            dim_mults = (1, 2, 4, 8),
            channels=4
        )
    else:
        simulator_model = Unet(
            dim = args.image_size,
            out_dim = 3,
            dim_mults = (1, 2, 4, 8),
            channels=6
        )
    force_model.load_state_dict(torch.load(args.force_model_checkpoint, map_location=lambda storage, loc: storage))
    simulator_model.load_state_dict(torch.load(args.simulator_model_checkpoint, map_location=lambda storage, loc: storage))
    boundary_updater_model.load_state_dict(torch.load(args.boundary_updater_model_checkpoint, map_location=lambda storage, loc: storage))
    force_model = force_model.to(device)
    force_model.eval()
    simulator_model = simulator_model.to(device)
    simulator_model.eval()
    boundary_updater_model = boundary_updater_model.to(device)
    boundary_updater_model.eval()
    ppl = SurrogatePipeline(
        simulator_model,
        force_model,
        boundary_updater_model
    ) 
    
    return ppl

def build_ppl(args):
    device = get_device(args.gpu)
    force_model = ForceUnet(
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        channels=4
    )
    force_model.load_state_dict(torch.load(args.force_model_checkpoint))
    simulator_model = Unet(
        dim = 128,
        out_dim = 3,
        dim_mults = (1, 2, 4, 8),
        channels=6
    )
    simulator_model.load_state_dict(torch.load(args.simulator_model_checkpoint))
    boundary_updater_model = Unet(
        dim = 128,
        out_dim = 3,
        dim_mults = (1, 2, 4, 8),
        channels=3
    )
    boundary_updater_model.load_state_dict(torch.load(args.boundary_updater_model_checkpoint))
    force_model = force_model.to(device)
    force_model.eval()
    simulator_model = simulator_model.to(device)
    simulator_model.eval()
    boundary_updater_model = boundary_updater_model.to(device)
    boundary_updater_model.eval()
    ppl = SurrogatePipeline(
        simulator_model,
        force_model,
        boundary_updater_model
    ) 
    
    return ppl


def load_data(args):
    dataset = Jellyfish(
        dataset="jellyfish", 
        dataset_path=args.dataset_path,
        time_steps=args.time_steps, 
        steps=20, 
        time_interval=1, 
        is_train=True, 
        show_missing_files=False, 
        is_traj=False, 
        is_testdata=args.is_testdata,
        for_pipeline=True
    )
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, num_workers = 32)
    
    return train_loader

def control_model(force_x, theta_t, theta_T, args):
    """
    TODO
    Zhang Tao and Hu Pei yan write control model here
    """

    return theta_t + 0.01
    
            
def main(args):
    ppl = build_ppl(args)
    dataloader = load_data(args)
    device = get_device(args.gpu)
    print("number of batch in train_loader: ", len(dataloader))
    for i, data in enumerate(dataloader):
        print("i: ", i)
        state, mask_offsets, thetas, bd_mask_offset_0, time_id = data
        thetas = thetas.to(device)
        state_pad = torch.zeros(state.shape[0], args.frames, 3, args.image_size, args.image_size).to(device)
        mask_offsets_pad = torch.zeros(mask_offsets.shape[0], args.frames, 3, args.image_size, args.image_size).to(device)
        bd_mask_offset_0_pad = torch.zeros(bd_mask_offset_0.shape[0], 3, args.image_size, args.image_size).to(device)
        state_pad[:,:,:,1:-1,1:-1] = state.to(device)
        mask_offsets_pad[:,:,:,1:-1,1:-1] = mask_offsets.to(device)
        bd_mask_offset_0_pad[:,:,1:-1,1:-1] = bd_mask_offset_0.to(device)
        theta_t, theta_T = thetas[:,0], thetas[:,-1]
        state_t = state_pad[:,0,:,:,:]
        mask_offsets_t = mask_offsets_pad[:,0,:,:,:]
        for t in range(args.frames):
            state_t, force_t = ppl.run(state_t, mask_offsets_t, theta_t) # output state and force of next time step
            theta_t = control_model(force_t, theta_t, theta_T, args) # output theta of next time step by control model
            p.print(f"self.ppl.update_mask_offsets start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
            mask_offsets_t = ppl.update_mask_offsets(bd_mask_offset_0_pad, theta_t) # output mask_offsets of next time step
            p.print(f"self.ppl.update_mask_offsets end", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="path", type=str, help='path to dataset') 
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--time_steps', type=int, default=40, help='number of all time steps in each simulation')
    parser.add_argument('--frames', type=int, default=20, help='number of time steps in each training sample')
    parser.add_argument('--m', type=int, default=20, help='number of points on each ellipse wing of jellyfish')
    parser.add_argument('--force_model_checkpoint', type=str, default='path')
    parser.add_argument('--simulator_model_checkpoint', type=str, default='path')
    parser.add_argument('--boundary_updater_model_checkpoint', type=str, default='path')
    parser.add_argument('--is_testdata', default=False, type=bool,
                    help='whether run mini example data, if True, yes; otherwise, run full data')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id used in training')
    args = parser.parse_args()
    
    main(args)