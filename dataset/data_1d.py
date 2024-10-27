from .apps.burgers_h5py import Burgers
import torch
import torch.nn as nn


class Burgers1D(Burgers):
    '''Burgers dataset compatible with the DDPM Trainer class.
    '''
    def __init__(
            self, 
            *args, 
            device='cpu', 
            rescaler=None, 
            stack_u_and_f=False, 
            pad_for_2d_conv=False, 
            partially_observed_fill_zero_unobserved=None, 
            **kwargs
    ): 
        super().__init__(*args, **kwargs)
        self.device = device
        
        if rescaler is None:    
            self.calculate_rescaler()
        else:
            self.rescaler = rescaler
        
        self.stack_u_and_f = stack_u_and_f
        self.pad_for_2d_conv = pad_for_2d_conv
        self.fill_zero_unobserved = partially_observed_fill_zero_unobserved

    def calculate_rescaler(self):
        '''Rescale every data to (0, 1)'''
        u = torch.tensor([x[1] for x in self.dataset_cache])
        f = torch.tensor([x[2] for x in self.dataset_cache])
        self.rescaler = torch.cat((u.squeeze(), f), dim=1).abs().max()


    def get(self, idx, use_normalized=True):
        '''We are only returning f and u for now, in the shape of 
        (u0, u1, ..., f0, f1, ...)
        This layout should not make a difference if f is not 
        '''
        # note that we need to get sim_id in case we split different time intervals
        # in the same run as different samples
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        _, data_traj, force, x_pos = self.dataset_cache[sim_id]

        u = torch.tensor(data_traj, dtype=torch.float32)
        f = torch.tensor(force, dtype=torch.float32)
        
        # we assume that u.size() == (Nt, Nx, 1) and 
        # f.size() == (Nt (or 1 if f is const), Nx, 1)
        
        if self.fill_zero_unobserved is not None:
            if self.fill_zero_unobserved == 'front_rear_quarter':
                u = u.squeeze()
                nx = u.shape[1]
                u[:, nx // 4: (nx * 3) // 4] = 0
            else:
                raise ValueError('Unknown partially observed mode')
        
        if self.stack_u_and_f:
            assert self.pad_for_2d_conv
            # pad f for stack
            nt = f.size(0)
            f = nn.functional.pad(f, (0, 0, 0, 16 - nt), 'constant', 0)
            u = nn.functional.pad(u.squeeze(), (0, 0, 0, 15 - nt), 'constant', 0)
            
            data = torch.stack((u, f), dim=0)
        else: # for get_target
            assert not self.pad_for_2d_conv
            data = torch.cat((u.squeeze(), f), dim=0).squeeze()

        if use_normalized:
            data = data / self.rescaler
        
        return data
