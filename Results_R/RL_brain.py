import copy
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        """Initialization."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ):
        """Store the transition in buffer."""
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
            self,
            size: int,
            mu: float = 0.0,
            theta: float = 0.15,
            sigma: float = 0.2
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (=noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state


class Actor(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            init_w: float = 3e-3
    ):
        """Initialization."""
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)

        # 修改输出层
        self.fc3_R = nn.Linear(128, 1)  # 第一个动作参数的输出层
        self.fc3_B = nn.Linear(128, 1)  # 第二个动作参数的输出层

        self.fc3_R.weight.data.uniform_(-init_w, init_w)
        self.fc3_R.bias.data.uniform_(-init_w, init_w)

        self.fc3_B.weight.data.uniform_(-init_w, init_w)
        self.fc3_B.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        u_R = torch.tanh(self.fc3_R(x))  # 第一个动作参数，使用 tanh 激活函数
        u_B = torch.tanh(self.fc3_B(x))  # 第二个动作参数，使用 tanh 激活函数
        return u_R, u_B


class Critic(nn.Module):
    def __init__(
            self,
            in_dim: int,
            init_w: float = 3e-3
    ):
        """Initialization."""
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(
            self,
            state: torch.Tensor,
            action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.fc3(x)

        return value


class DDPG:
    def __init__(
            self,
            obs_dim,
            action_dim: int,
            memory_size: int,
            batch_size: int,
            ou_noise_theta: float,
            ou_noise_sigma: float,
            gamma: float = 1,
            tau: float = 5e-4,
            initial_random_steps: int = 1e4,
            gamma1: float = 1.5,
            v_max: int = 300

    ):
        """Initialization."""
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps

        self.gamma1 = gamma1

        self.v_max = v_max
        self.initial_u_R = self.gamma1 - 1

        self.initial_u_B = 0.3
        # noise
        self.noise = OUNoise(
            action_dim,
            theta=ou_noise_theta,
            sigma=ou_noise_sigma
        )

        # device
        self.device = torch.device('cuda')
        print(self.device)

        # networks
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target = Critic(obs_dim + action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.writer = SummaryWriter('tensorboard')

        # transition to store in memory
        self.transition = list()

        # total steps count
        self.total_step = 0

        # mode
        self.is_test = False

        log_mult = []
        u_R = self.initial_u_R if self.initial_u_R > 0 else np.random.uniform(0, 1)
        tensor = torch.nn.Parameter(torch.tensor(u_R, dtype=torch.float32).to(self.device), requires_grad=True)
        log_mult.append(tensor)
        u_B = self.initial_u_B if self.initial_u_B > 0 else np.random.uniform(0, 1)
        tensor1 = torch.nn.Parameter(torch.tensor(u_B, dtype=torch.float32).to(self.device), requires_grad=True)
        log_mult.append(tensor1)
        self.log_multiplier = log_mult

        # 优化算法
        # self.multiplier_optimizer = torch.optim.Adam([x for x in log_mult if x is not None], lr=3e-4)
        self.log_multiplier = torch.tensor(log_mult, dtype=torch.float32).to(self.device)

        # Optimizer for multiplier parameters
        self.multiplier_optimizer = torch.optim.Adam([self.log_multiplier], lr=3e-4)

    def select_action(self, state: np.ndarray):
        # Convert state to tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get action parameters from the actor network
        action_params = self.actor(state_tensor)

        # Unpack action parameters and select the maximum value
        u_R, u_B = action_params
        selected_action = max(u_R.item(), u_B.item())

        # Add noise to the selected action
        noise = self.noise.sample()
        selected_action = np.clip(selected_action + noise[0], -0.99, 0.99)  # Selecting noise[0] for the first element

        # Store state and action parameters
        self.transition = [state, selected_action]

        print(selected_action.shape)
        return selected_action

    def update_multiplier(self, state: np.ndarray, p_t: float, v_t: float, rho: float):
        # Convert state to tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Get action parameters from the actor network
        action_params = self.actor(state_tensor)

        # Convert the action parameters tuple to a list
        u_R, u_B = action_params

        # Calculate roi_diff and rbr
        roi_diff = torch.tensor(v_t / (1e-3 + p_t) - self.gamma1, dtype=torch.float32).to(self.device)
        rbr = torch.tensor(rho - p_t, dtype=torch.float32).to(self.device)

        # Calculate loss for multiplier parameters
        multiplier_loss = torch.tensor(0., dtype=torch.float32).to(self.device)
        if u_R is not None:
            multiplier_loss += u_R * roi_diff.mean()

        if u_B is not None:
            multiplier_loss += u_B * rbr.mean()

        # Backpropagation and optimization
        self.actor_optimizer.zero_grad()
        multiplier_loss.backward()
        self.actor_optimizer.step()

    def store(self, reward: float, next_state: np.ndarray, done: bool):
        # Add reward, next state, and done flag to the transition
        self.transition += [reward, next_state, done]

        # Store the transition in the replay buffer
        self.memory.store(*self.transition)

    def update_model(self) -> Tuple[float, float]:
        # Sample a batch of transitions from the replay buffer
        samples = self.memory.sample_batch()
        state = torch.as_tensor(samples["obs"], dtype=torch.float32).to(self.device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32).to(self.device)
        action_params = torch.as_tensor(samples["acts"], dtype=torch.float32).to(self.device)
        reward = torch.as_tensor(samples["rews"], dtype=torch.float32).to(self.device)
        done = torch.as_tensor(samples["done"], dtype=torch.float32).to(self.device)

        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks

        # Train critic
        values = self.critic(state, action_params)
        critic_loss = F.mse_loss(values, curr_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target network update
        self._target_soft_update()

        return actor_loss.item(), critic_loss.item()

    def _target_soft_update(self):
        tau = self.tau

        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1 - tau) * t_param.data)
