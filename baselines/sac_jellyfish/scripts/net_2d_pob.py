import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

# theta:0.18~1.05
LOG_STD_MAX = -2
LOG_STD_MIN = -5
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
            nn.Conv2d(8, 16, 5, stride=2, padding=2), 
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
            nn.Conv2d(8, 16, 5, stride=2, padding=2), 
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
        action = action.reshape(action.shape[0], 1, 1, 1).repeat(1, 1, self.num_inputs, self.num_inputs)
        xu = torch.cat([state, action], 1)
        x1 = self.Q1(xu)
        x2 = self.Q2(xu)
        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        activation = nn.ELU() 

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 16, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(32, 64, 5, stride=2, padding=2), 
            activation,
            nn.AvgPool2d(6),
            activation,
            nn.Flatten()
        )

        self.mean_conv = nn.Sequential(
                                        nn.Linear(64, 64),
                                        activation,
                                        nn.Linear(64, 1)
                                    )
        self.log_std_conv = nn.Sequential(
                                        nn.Linear(64, 64),
                                        activation,
                                        nn.Linear(64, 1)
                                    )

        self.mean_conv.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(0.45)
            self.action_bias = torch.tensor(0.6)
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
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()

        activation = nn.LeakyReLU()

        self.encoder = nn.Sequential(
            nn.Conv2d(9, 14, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, stride=2, padding=2), 
            activation,
            nn.Conv2d(32, 16, 5, padding=2),
            activation,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
            nn.Conv2d(16, 4, 5, padding=2)
        )

        self.mean_conv = nn.Sequential(nn.Conv2d(4, 1, 5, padding=2),
                                        nn.Flatten(),
                                        nn.Linear(num_inputs**2, hidden_dim),
                                        nn.Linear(hidden_dim, num_actions)
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


