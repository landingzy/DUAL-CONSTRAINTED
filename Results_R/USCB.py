import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import numpy as np


class Critic(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        self.FC1 = nn.Linear(self.dim_observation, 10)
        self.FC2 = nn.Linear(10 + dim_action, 50)
        self.FC3 = nn.Linear(50, 10)
        self.FC4 = nn.Linear(10, 1)

    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        result = F.relu(self.FC3(result))
        return self.FC4(result)


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 100)
        self.FC2 = nn.Linear(100, 10)
        self.FC3 = nn.Linear(10, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))
        return (result + 1) * 5


class USCB:
    def __init__(self, dim_obs,
                 dim_actions,
                 buffer_size: int,
                 sample_size: int,
                 gamma:float=1,
                 tau:float=0.1,
                 critic_lr:float=0.0001,
                 actor_lr:float=0.0001,

                 network_random_seed=1):

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.num_of_states = dim_obs
        self.num_of_actions = dim_actions

        torch.random.manual_seed(network_random_seed)
        # actors and critics and their targets
        self.actors = Actor(self.num_of_states, self.num_of_actions)
        self.critics = Critic(self.num_of_states, self.num_of_actions)

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.GAMMA = gamma
        self.tau = tau
        self.num_of_steps = 0

        self.var = 1
        self.critic_optimizer = Adam(self.critics.parameters(), lr=critic_lr)
        self.actor_optimizer = Adam(self.actors.parameters(), lr=actor_lr)

        self.num_of_episodes = 0

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.actors.cuda()
            self.critics.cuda()
            self.actors_target.cuda()
            self.critics_target.cuda()
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # replay buffer
        self.replay_buffer = {"states": np.zeros((buffer_size, self.num_of_states)),
                              "actions": np.zeros((buffer_size, self.num_of_actions)),
                              "rewards": np.zeros((buffer_size, 1)),
                              "next_states": np.zeros((buffer_size, self.num_of_states)),
                              "terminal": np.zeros((buffer_size, 1)),
                              "values": np.zeros((buffer_size, 1))}
        self.buffer_pointer = 0
        self.if_full = False
        self.buffer_size = buffer_size
        self.sample_size = sample_size

    def store_experience(self, states, actions, rewards, next_states, terminal, values):
        self.replay_buffer["states"][self.buffer_pointer] = deepcopy(states)
        self.replay_buffer["actions"][self.buffer_pointer] = deepcopy(actions)
        self.replay_buffer["rewards"][self.buffer_pointer] = deepcopy(rewards)
        self.replay_buffer["next_states"][self.buffer_pointer] = deepcopy(next_states)
        self.replay_buffer["terminal"][self.buffer_pointer] = deepcopy(terminal)
        self.replay_buffer["values"][self.buffer_pointer] = deepcopy(values)

        self.buffer_pointer += 1
        if self.buffer_pointer == self.buffer_size:
            self.if_full = True
            self.buffer_pointer = 0

    def sample_experience(self):
        sample_index_list = np.random.choice(self.buffer_size, self.sample_size, replace=False)
        experience_samples = np.zeros((self.sample_size, self.num_of_states + self.num_of_actions + 1 + self.num_of_states + 1 + 1))
        index = 0
        for i in sample_index_list:
            experience_samples[index, 0:self.num_of_states] = deepcopy(self.replay_buffer["states"][i])
            experience_samples[index, self.num_of_states:self.num_of_states + self.num_of_actions] = deepcopy(self.replay_buffer["actions"][i])
            experience_samples[index, self.num_of_states + self.num_of_actions] = deepcopy(self.replay_buffer["rewards"][i])
            experience_samples[index, self.num_of_states + self.num_of_actions + 1:self.num_of_states + self.num_of_actions + 1 + self.num_of_states] = deepcopy(self.replay_buffer["next_states"][i])
            experience_samples[index, self.num_of_states + self.num_of_actions + 1 + self.num_of_states] = deepcopy(self.replay_buffer["terminal"][i])
            experience_samples[index, self.num_of_states + self.num_of_actions + 1 + self.num_of_states + 1] = deepcopy(self.replay_buffer["values"][i])
            index += 1
        return experience_samples

    def train(self):
        if self.if_full:
            samples = torch.Tensor(self.sample_experience()).type(self.FloatTensor)
            states = samples[:, 0:self.num_of_states]
            actions = samples[:, self.num_of_states:(self.num_of_states + self.num_of_actions)]
            actions = actions / 5 - 1
            rewards = samples[:, self.num_of_actions + self.num_of_states].unsqueeze(1)
            next_states = samples[:, self.num_of_states + self.num_of_actions + 1:self.num_of_states + self.num_of_actions + 1 + self.num_of_states]
            terminal = samples[:, self.num_of_states + self.num_of_actions + 1 + self.num_of_states]
            values = samples[:, self.num_of_states + self.num_of_actions + 1 + self.num_of_states + 1].unsqueeze(1)

            for i in range(10):
                current_Q = self.critics(states, actions)
                loss_Q = torch.nn.MSELoss()(current_Q, values)
                self.critic_optimizer.zero_grad()
                loss_Q.backward()
                self.critic_optimizer.step()

            for i in range(1):
                actions_this_agent = self.actors(states)
                actions_this_agent = actions_this_agent / 5 - 1
                loss_A = -self.critics(states, actions_this_agent)
                loss_A = loss_A.mean()
                self.actor_optimizer.zero_grad()
                loss_A.backward()
                self.actor_optimizer.step()

            return loss_Q.cpu().data.numpy(), loss_A.cpu().data.numpy()
        else:
            return None, None

    def take_actions(self, states, mode="behavior"):
        states = torch.Tensor(states).type(self.FloatTensor)
        actions = self.actors(states)
        actions = actions.cpu().data.numpy()
        if mode == "behavior":
            actions += np.random.randn(1) * 0.1
            actions = actions if actions >= 0 else 0
        elif mode == "target":
            pass
        return actions

    def update_target(self):
        for target_param, source_param in zip(self.critics_target.parameters(), self.critics.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)
        for target_param, source_param in zip(self.actors_target.parameters(), self.actors.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    def save_net(self, save_path):
        try:
            torch.save(self.critics, save_path + "/critic.pkl")
            torch.save(self.actors, save_path + "/actor.pkl")
        except:
            print("save net failed: there is no such path")

    def load_net(self, load_path="saved_model/fixed_initial_budget", iteration=None):
        try:
            if iteration is None:
                self.critics = torch.load(load_path + "/critic.pkl", map_location=torch.device('cpu'))
                self.actors = torch.load(load_path + "/actor.pkl", map_location=torch.device('cpu'))
            else:
                self.critics = torch.load(load_path + "/critic_level_" + str(iteration) + ".pkl", map_location=torch.device('cpu'))
                self.actors = torch.load(load_path + "/actor_level_" + str(iteration) + ".pkl", map_location=torch.device('cpu'))

            self.actors_target = deepcopy(self.actors)
            self.critics_target = deepcopy(self.critics)
            self.critic_optimizer = Adam(self.critics.parameters(), lr=self.critic_lr)
            self.actor_optimizer = Adam(self.actors.parameters(), lr=self.actor_lr)
            self.use_cuda = torch.cuda.is_available()
            if self.use_cuda:
                self.actors.cuda()
                self.critics.cuda()
                self.actors_target.cuda()
                self.critics_target.cuda()
        except:
            print("load net failed: there is no such path")
