import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MAX = 1
LOG_STD_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

        
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.num_inputs = num_inputs
        activation = nn.ELU()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            activation,
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            activation,
            nn.AvgPool2d(3),
            activation,
            nn.Flatten(),
            nn.Linear(128, 128),
            activation,
            nn.Linear(128, 1)
        )

        # Q2 architecture
        self.Q2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            activation,
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            activation,
            nn.AvgPool2d(3),
            activation,
            nn.Flatten(),
            nn.Linear(128, 128),
            activation,
            nn.Linear(128, 1)
        )

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = self.Q1(xu)
        x2 = self.Q2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        activation = nn.ELU() 

        self.encoder = nn.Sequential(
            nn.Conv2d(num_inputs, 8, 5, padding=2), 
            activation,
            nn.Conv2d(8, 16, 5, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, padding=2), 
        )

        self.mean_conv = nn.Sequential(
                                nn.Conv2d(32, 8, 5, padding=2), 
                                activation,
                                nn.Conv2d(8, num_actions, 5, padding=2), 
                                    )
        
        self.log_std_conv = nn.Sequential(
                                nn.Conv2d(32, 8, 5, padding=2), 
                                activation,
                                nn.Conv2d(8, num_actions, 5, padding=2), 
                                    )

        self.mean_conv.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1)
            self.action_bias = torch.tensor(0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.encoder(state)
        mean = self.mean_conv(x)
        log_std = self.log_std_conv(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        mean[:,:,8:56,8:56] = 0 # indirect control
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        action[:,:,8:56,8:56] = 0 # indirect control
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True).mean((-1,-2))
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# NOT IMPLEMENTED
class DeterministicPolicy(nn.Module): 
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        activation = nn.LeakyReLU()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_inputs, 8, 5, padding=2), 
            activation,
            nn.Conv2d(8, 16, 5, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, padding=2), 
        )

        self.mean_conv = nn.Sequential(
                                nn.Conv2d(32, 8, 5, padding=2), 
                                activation,
                                nn.Conv2d(8, num_actions, 5, padding=2), 
                                    )
        
        self.log_std_conv = nn.Sequential(
                                nn.Conv2d(32, 8, 5, padding=2), 
                                activation,
                                nn.Conv2d(8, num_actions, 5, padding=2), 
                                    )

        self.noise = torch.Tensor(num_actions, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.encoder(state)
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1).to(mean.device)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean


