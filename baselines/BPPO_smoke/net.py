import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple

LOG_STD_MAX = -2
LOG_STD_MIN = -5
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_clamp(
    x: torch.Tensor, bound: tuple
    ) -> torch.Tensor:
    low, high = bound
    x = low + 0.5 * (high - low) * (x + 1)
    return x


def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    final_activation: str
) -> torch.nn.modules.container.Sequential:

    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)



class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, state_dim: int, hidden_dim: int, depth: int
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, 1, 'tanh')

    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._net(s)
    

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, channel):
        super(ValueNetwork, self).__init__()
        self.num_inputs = num_inputs
        activation = nn.ELU()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Conv2d(channel-1, 16, 5, stride=2, padding=2), 
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
    
    def forward(self, state):
        xu = state
        x1 = self.Q1(xu)
        return x1

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, channel):
        super(QNetwork, self).__init__()
        self.num_inputs = num_inputs
        activation = nn.ELU()

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Conv2d(channel+1, 16, 5, stride=2, padding=2), 
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
        action = action.reshape(action.shape[0], 2, 64, 64)
        xu = torch.cat([state, action], 1)
        x1 = self.Q1(xu)
        return x1

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, channel, action_space=None):
        super(GaussianPolicy, self).__init__()

        activation = nn.ELU() 

        self.encoder = nn.Sequential(
            nn.Conv2d(channel-1, 16, 5, stride=1, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, stride=1, padding=2), 
            activation,
            nn.Conv2d(32, 32, 5, stride=1, padding=2), 
        )

        self.mean_conv = nn.Sequential(
                                        nn.Conv2d(32, 16, 5, stride=1, padding=2), 
                                        activation,
                                        nn.Conv2d(16, 8, 5, stride=1, padding=2), 
                                        activation,
                                        nn.Conv2d(8, 2, 5, stride=1, padding=2)
                                    )
        self.log_std_conv = nn.Sequential(
                                        nn.Conv2d(32, 16, 5, stride=1, padding=2), 
                                        activation,
                                        nn.Conv2d(16, 8, 5, stride=1, padding=2), 
                                        activation,
                                        nn.Conv2d(8, 2, 5, stride=1, padding=2)
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
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist

    def sample(self, state):
        mean, log_std = self.forward(state)
        
        mean[:,:,8:56,8:56] = 0
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    

class GaussianPolicy_2(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, channel, action_space=None):
        super(GaussianPolicy_2, self).__init__()
        
        activation = nn.ELU() 

        self.encoder = nn.Sequential(
            nn.Conv2d(channel-1, 16, 5, stride=1, padding=2), 
            activation,
            nn.Conv2d(16, 32, 5, stride=1, padding=2), 
            activation,
            nn.Conv2d(32, 2, 5, stride=1, padding=2), 
            activation,
        )
        
        self.mean_conv = nn.Sequential(
                                        nn.Linear(64, 64),
                                        activation,
                                        nn.Linear(64, 64)
                                    )
        self.log_std_conv = nn.Sequential(
                                        nn.Linear(64, 64),
                                        activation,
                                        nn.Linear(64, 64)
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
        std = log_std.exp()
        dist = Normal(mean, std)
        return dist

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob = log_prob - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class GaussPolicyMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential
    _log_std_bound: tuple

    def __init__(
        self, 
        state_dim: int, hidden_dim: int, depth: int, action_dim: int, 
    ) -> None:
        super().__init__()
        self._net = MLP(state_dim, hidden_dim, depth, (2 * action_dim), 'tanh')
        self._log_std_bound = (-5., 0.)


    def forward(
        self, s: torch.Tensor
    ) -> torch.distributions:

        mu, log_std = self._net(s).chunk(2, dim=-1)
        log_std = soft_clamp(log_std, self._log_std_bound)
        std = log_std.exp()

        dist = Normal(mu, std)
        return dist
