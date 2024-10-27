import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
        
class Net_f(nn.Module):
    def __init__(self, h):
        super(Net_f, self).__init__()
        self.h = h // 4
        
        self.down = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(8, 16, 5, stride=2, padding=2), 
            nn.Tanh(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), 
            nn.Tanh(),
            nn.Conv1d(32, 8, 5, padding=2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(8*self.h, 8*self.h)
        )

        self.up = nn.Sequential(
            nn.Conv1d(8, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(32, 16, 5, padding=2),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(16, 1, 5, padding=2)
        )
        # self.apply(weights_init_)
        
    def forward(self, f):
        f_latent = self.down(f)
        next_f = f_latent.view(-1,8,self.h)
        f_rec = self.up(next_f)
        
        return f_rec, f_latent
    
class Net_u(nn.Module):
    def __init__(self, h):
        super(Net_u, self).__init__()
        self.h = h // 4
        
        self.down = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(8, 16, 5, stride=2, padding=2), 
            nn.Tanh(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), 
            nn.Tanh(),
            nn.Conv1d(32, 8, 5, padding=2),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(8*self.h, 8*self.h)
        )

        self.up = nn.Sequential(
            nn.Conv1d(8, 64, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(32, 16, 5, padding=2),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True), 
            nn.Conv1d(16, 1, 5, padding=2)
        )
        # self.apply(weights_init_)
        
    def forward(self, u):
        u_latent = self.down(u)
        next_u = u_latent.view(-1,8,self.h)
        u_rec = self.up(next_u)
        
        return u_rec, u_latent
    
class Net_trans(nn.Module):
    def __init__(self):
        super(Net_trans, self).__init__()
        self.transition1=nn.Conv1d(16, 8, 3, padding=1)
        # self.apply(weights_init_)

    def forward(self, x):
        x=self.transition1(x)
        return x