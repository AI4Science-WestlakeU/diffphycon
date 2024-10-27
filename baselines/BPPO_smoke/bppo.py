import torch
import numpy as np
from buffer import OnlineReplayBuffer
from net import GaussPolicyMLP, GaussianPolicy
from critic import ValueLearner, QLearner
from ppo import ProximalPolicyOptimization
from utils_RL import CONST_EPS, log_prob_func, orthogonal_initWeights


class BehaviorCloning:
    _device: torch.device
    _policy: GaussianPolicy
    _optimizer: torch.optim
    _policy_lr: float
    _batch_size: int
    _channel: int
    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        batch_size: int,
        channel: int
    ) -> None:
        super().__init__()
        self._device = device
        self._policy = GaussianPolicy(state_dim, action_dim, state_dim, channel).to(device)
        orthogonal_initWeights(self._policy)
        self._optimizer = torch.optim.Adam(
            self._policy.parameters(),
            lr = policy_lr
        )
        self._lr = policy_lr
        self._batch_size = batch_size
        self.channel = channel
        

    def loss(
        self, memory: OnlineReplayBuffer,
    ) -> torch.Tensor:
        state_batch, action_batch, next_state_batch, mask_batch = next(memory)
        reward_batch = next_state_batch[:,3].mean((-1,-2))
        state_batch, next_state_batch = state_batch[:,[0,1,2,4]], next_state_batch[:,[0,1,2,4]]
        state_batch = state_batch.to(self._device).to(torch.float32)
        next_state_batch = next_state_batch.to(self._device).to(torch.float32)
        action_batch = action_batch.to(self._device).to(torch.float32)
        reward_batch = reward_batch.float().to(self._device).unsqueeze(1).to(torch.float32)
        
        dist = self._policy(state_batch)
        
        log_prob = log_prob_func(dist, action_batch) 
        loss = (-log_prob).mean()
        
        return loss


    def update(
        self, replay_buffer: OnlineReplayBuffer,
        ) -> float:
        policy_loss = self.loss(replay_buffer)

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        return policy_loss.item()


    def select_action(
        self, s: torch.Tensor, is_sample: bool
    ) -> torch.Tensor:
        dist = self._policy(s)
        if is_sample:
            action = dist.sample()
        else:    
            action = dist.mean
        # clip 
        action = action.clamp(-1., 1.)
        return action


    def offline_evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10
        ) -> float:
        env = gym.make(env_name)
        env.seed(seed)

        total_reward = 0
        for _ in range(eval_episodes):
            s, done = env.reset(), False
            while not done:
                s = torch.FloatTensor((np.array(s).reshape(1, -1) - mean) / std).to(self._device)
                a = self.select_action(s, is_sample=False).cpu().data.numpy().flatten()
                s, r, done, _ = env.step(a)
                total_reward += r
        
        avg_reward = total_reward / eval_episodes
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        return d4rl_score
    

    def save(
        self, path: str
    ) -> None:
        torch.save(self._policy.state_dict(), path)
        print('Behavior policy parameters saved in {}'.format(path))
    

    def load(
        self, path: str
    ) -> None:
        self._policy.load_state_dict(torch.load(path, map_location=self._device))
        print('Behavior policy parameters loaded')



class BehaviorProximalPolicyOptimization(ProximalPolicyOptimization):

    def __init__(
        self,
        device: torch.device,
        state_dim: int,
        hidden_dim: int, 
        depth: int,
        action_dim: int,
        policy_lr: float,
        clip_ratio: float,
        entropy_weight: float,
        decay: float,
        omega: float,
        batch_size: int,
        channel: int
    ) -> None:
        super().__init__(
            device = device,
            state_dim = state_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            action_dim = action_dim,
            policy_lr = policy_lr,
            clip_ratio = clip_ratio,
            entropy_weight = entropy_weight,
            decay = decay,
            omega = omega,
            batch_size = batch_size,
            channel = channel)
        self.channel = channel


    def loss(
        self, 
        memory: OnlineReplayBuffer,
        Q: QLearner,
        value: ValueLearner,
        is_clip_decay: bool,
    ) -> torch.Tensor:
        # -------------------------------------Advantage-------------------------------------        
        state_batch, action_batch, next_state_batch, mask_batch = next(memory)
        reward_batch = next_state_batch[:,3].mean((-1,-2))
        state_batch, next_state_batch = state_batch[:,[0,1,2,4]], next_state_batch[:,[0,1,2,4]]
        state_batch = state_batch.to(self._device).to(torch.float32)
        next_state_batch = next_state_batch.to(self._device).to(torch.float32)
        action_batch = action_batch.to(self._device).to(torch.float32)
        reward_batch = reward_batch.float().to(self._device).unsqueeze(1).to(torch.float32)
        
        old_dist = self._old_policy(state_batch)
        a = old_dist.rsample()
        
        advantage = Q(state_batch, a) - value(state_batch)
        advantage = (advantage - advantage.mean()) / (advantage.std() + CONST_EPS)
        # -------------------------------------Advantage-------------------------------------
        new_dist = self._policy(state_batch)

        new_log_prob = log_prob_func(new_dist, a)
        old_log_prob = log_prob_func(old_dist, a)
        ratio = (new_log_prob - old_log_prob).exp()
                
        advantage = self.weighted_advantage(advantage)
        
        loss1 =  ratio * advantage 
        
        if is_clip_decay:
            self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio

        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage         
        entropy_loss = new_dist.entropy().sum((-1,-2,-3), keepdim=True) * self._entropy_weight
        
        
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()

        return loss


    def offline_evaluate(
        self,
        env_name: str,
        seed: int,
        mean: np.ndarray,
        std: np.ndarray,
        eval_episodes: int=10
        ) -> float:
        env = gym.make(env_name)
        avg_reward = self.evaluate(env_name, seed, mean, std, eval_episodes)
        d4rl_score = env.get_normalized_score(avg_reward) * 100
        return d4rl_score
