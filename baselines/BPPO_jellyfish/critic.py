import torch
import torch.nn.functional as F

from net import ValueMLP, ValueNetwork, QNetwork
from buffer import OnlineReplayBuffer


class ValueLearner:
    _device: torch.device
    _value: ValueNetwork
    _optimizer: torch.optim
    _batch_size: int
    _channel: int

    def __init__(
        self, 
        device: torch.device, 
        state_dim: int, 
        value_lr: float, 
        batch_size: int,
        channel: int
    ) -> None:
        super().__init__()
        self._device = device
        self.channel = channel
        self._value = ValueNetwork(state_dim, channel).to(device)
        self._optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=value_lr,
            )
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        return self._value(s)


    def update(
        self, memory: OnlineReplayBuffer, reward_alpha, reg_lambda
    ) -> float:
        
        state_batch, action_batch, force_batch, cond_T_batch, reg_batch, next_state_batch, mask_batch = next(memory)
        reward_batch = reward_alpha * force_batch + (-1) * cond_T_batch + 20 * reward_alpha * reg_lambda * reg_batch

        if self.channel==10:
            state_batch = torch.FloatTensor(state_batch[:,:]).to(self._device)
            next_state_batch = torch.FloatTensor(next_state_batch[:,:]).to(self._device)
        elif self.channel==8:
            state_batch = torch.FloatTensor(state_batch[:,2:]).to(self._device)
            next_state_batch = torch.FloatTensor(next_state_batch[:,2:]).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = reward_batch.float().to(self._device).unsqueeze(1)
        value_loss = F.mse_loss(self._value(state_batch), reward_batch)
        
        self._optimizer.zero_grad()
        value_loss.backward()
        self._optimizer.step()

        return value_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._value.state_dict(), path)
        print('Value parameters saved in {}'.format(path))


    def load(
        self, path: str
    ) -> None:
        self._value.load_state_dict(torch.load(path, map_location=self._device))
        print('Value parameters loaded')

class QLearner:
    _device: torch.device
    _Q: QNetwork
    _optimizer: torch.optim
    _target_Q: QNetwork
    _total_update_step: int
    _target_update_freq: int
    _tau: float
    _gamma: float
    _batch_size: int
    _channel: int

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int,
        channel: int
    ) -> None:
        super().__init__()
        self._device = device
        self._Q = QNetwork(state_dim, action_dim, hidden_dim, channel).to(device)
        self._optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )

        self._target_Q = QNetwork(state_dim, action_dim, hidden_dim, channel).to(device)
        self._target_Q.load_state_dict(self._Q.state_dict())
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau

        self._gamma = gamma
        self._batch_size = batch_size


    def __call__(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> torch.Tensor:
        return self._Q(s, a)


    def loss(
        self, replay_buffer: OnlineReplayBuffer, reward_alpha, reg_lambda
    ) -> torch.Tensor:
        raise NotImplementedError


    def update(
        self, replay_buffer: OnlineReplayBuffer, reward_alpha, reg_lambda
    ) -> float:
        Q_loss = self.loss(replay_buffer, reward_alpha, reg_lambda)
        self._optimizer.zero_grad()
        Q_loss.backward()
        self._optimizer.step()

        self._total_update_step += 1
        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return Q_loss.item()


    def save(
        self, path: str
    ) -> None:
        torch.save(self._Q.state_dict(), path)
        print('Q function parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._Q.load_state_dict(torch.load(path, map_location=self._device))
        self._target_Q.load_state_dict(self._Q.state_dict())
        print('Q function parameters loaded')



class QSarsaLearner(QLearner):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int,
        channel: int
    ) -> None:
        super().__init__(
        device = device,
        state_dim = state_dim,
        action_dim = action_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        Q_lr = Q_lr,
        target_update_freq = target_update_freq,
        tau = tau,
        gamma = gamma,
        batch_size = batch_size,
        channel = channel
        )
        self.channel = channel


    def loss(
        self, memory: OnlineReplayBuffer, reward_alpha, reg_lambda
    ) -> torch.Tensor:
        
        state_batch, action_batch, force_batch, cond_T_batch, reg_batch, next_state_batch, mask_batch = next(memory)
        reward_batch = reward_alpha * force_batch + (-1) * cond_T_batch + 20 * reward_alpha * reg_lambda * reg_batch
        if self.channel==10:
            state_batch = torch.FloatTensor(state_batch[:,:]).to(self._device)
            next_state_batch = torch.FloatTensor(next_state_batch[:,:]).to(self._device)
        elif self.channel==8:
            state_batch = torch.FloatTensor(state_batch[:,2:]).to(self._device)
            next_state_batch = torch.FloatTensor(next_state_batch[:,2:]).to(self._device)
        action_batch = torch.FloatTensor(action_batch).to(self._device)
        reward_batch = reward_batch.float().to(self._device).unsqueeze(1)
        with torch.no_grad():
            target_Q_value = reward_batch + 1 * self._gamma * self._target_Q(next_state_batch, action_batch)
        
        Q = self._Q(state_batch, action_batch)
        
        loss = F.mse_loss(Q, target_Q_value)
        
        return loss



class QPiLearner(QLearner):
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        depth: int,
        Q_lr: float,
        target_update_freq: int,
        tau: float,
        gamma: float,
        batch_size: int
    ) -> None:
        super().__init__(
        device = device,
        state_dim = state_dim,
        action_dim = action_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        Q_lr = Q_lr,
        target_update_freq = target_update_freq,
        tau = tau,
        gamma = gamma,
        batch_size = batch_size
        )


    def loss(
        self, replay_buffer: OnlineReplayBuffer, pi
    ) -> torch.Tensor:
        s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        a_p = pi.select_action(s_p, is_sample=True)
        with torch.no_grad():
            target_Q_value = r + not_done * self._gamma * self._target_Q(s_p, a_p)
        
        Q = self._Q(s, a)
        loss = F.mse_loss(Q, target_Q_value)
        
        return loss