from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import torch
from torch.nn import functional as f

from per import Annealing
from src.model import Actor, Critic


class DDPGAgent:
    def __init__(self, state_size: int, action_size: int, whole_state_size: int,
                 whole_actions_size: int, gamma: float, idx: int,
                 soft_update_step: float, optimizers_cls: Dict, optimizers_kwargs: Dict,
                 device: torch.device, seed: int = 55321):
        self.device = device
        torch.manual_seed(seed)
        
        self.critic = Critic(state_size=whole_state_size, action_size=whole_actions_size,
                             seed=seed,
                             device=device, fcs1_units=128, fc2_units=100)
        self.critic_target = deepcopy(self.critic)
        
        self.actor = Actor(state_size=state_size, action_size=action_size, seed=seed,
                           device=device, fc1_units=156, fc2_units=156)
        self.actor_target = deepcopy(self.actor)
        self.actor_target.reset_parameters()
        
        self.actor_optim = optimizers_cls['actor'](self.actor.parameters(),
                                                   **optimizers_kwargs['actor'])
        self.critic_optim = optimizers_cls['critic'](self.critic.parameters(),
                                                     **optimizers_kwargs['critic'])
        
        self.index = idx
        self.gamma = gamma
        self.soft_update_step = soft_update_step
        self.noise = GaussianNoise(action_size, seed, sigma=0.25)
        self.noise_decay = Annealing(0.6, 0.05, 3e4)
    
    def save(self, folder, file_prefix):
        torch.save(self.actor.state_dict(), f"{folder}/{file_prefix}_actor_{self.index}.pth")
        torch.save(self.critic.state_dict(), f"{folder}/{file_prefix}_critic_{self.index}.pth")
    
    def load(self, folder, file_prefix):
        self.actor.load_state_dict(torch.load(f'{folder}/{file_prefix}_actor_{self.index}.pth'))
        self.critic.load_state_dict(torch.load(f'{folder}/{file_prefix}_critic_{self.index}.pth'))
    
    def act(self, state: np.ndarray, train: bool = True):
        state = torch.FloatTensor(state)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0).to(self.device)).cpu()
        self.actor.train()
        if train:
            action += self.noise.sample() * self.noise_decay.step()
        return torch.clamp(action, torch.tensor([-1, 0]), torch.tensor([1, 1])).numpy()
    
    def learn(self, transitions, whole_states, whole_next_states, whole_next_actions):
        agent_states, actions, rewards, next_states, dones = transitions
        next_values = self.critic_target(whole_next_states.flatten(1), whole_next_actions)
        
        targets_q = rewards + (self.gamma * next_values.flatten() * (1 - dones))
        
        expected_q = self.critic(whole_states.flatten(1), actions.flatten(1)).flatten()
        critic_loss = (expected_q - targets_q.detach()).squeeze()
        
        critic_loss = f.mse_loss(critic_loss, torch.zeros_like(critic_loss))
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        predicted_actions = self.actor(agent_states)
        whole_predicted_actions = actions.detach().swapaxes(0, 1)
        whole_predicted_actions[self.index] = predicted_actions
        whole_predicted_actions = whole_predicted_actions.swapaxes(0, 1).flatten(1)
        actor_loss = -self.critic(whole_states.flatten(1), whole_predicted_actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
    
    def get_target_actions(self, next_states):
        return self.actor_target(next_states)
    
    def reset_noise(self):
        self.noise.reset()
    
    def soft_update(self, local, target):
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(
                    self.soft_update_step * l_param.data + (
                            1 - self.soft_update_step) * t_param.data)


class MADDPG:
    def __init__(self, num_agents: int, action_size: int, state_size: int,
                 optimizers_cls: List[Dict], optimizers_kwargs: List[Dict], beta: Annealing,
                 alpha: float, batch_size: int, buffer_size: Union[int, float], gamma: float,
                 update_every: int, device: torch.device, soft_update_step: float, seed: int):
        self.agents = [DDPGAgent(action_size=action_size, state_size=state_size,
                                 whole_actions_size=action_size * num_agents,
                                 whole_state_size=state_size * num_agents, idx=n,
                                 soft_update_step=soft_update_step, gamma=gamma,
                                 optimizers_cls=optimizers_cls[n],
                                 optimizers_kwargs=optimizers_kwargs[n],
                                 seed=seed, device=device) for n in range(num_agents)]
        self.memory = ReplayBuffer(batch_size=batch_size, buffer_size=buffer_size, seed=seed,
                                   device=device, num_agents=num_agents)
        
        self.update_every = update_every
    
    def save(self, folder, prefix):
        [ag.save(folder, prefix) for ag in self.agents]
    
    def load(self, folder, file_prefix):
        [ag.load(folder, file_prefix) for ag in self.agents]
    
    def act(self, states, train=True):
        actions = np.array(
                [agent.act(state, train).reshape(-1) for state, agent in zip(states, self.agents)])
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        
        curr_step = self.memory.buffer_length
        if curr_step % self.update_every == 0 and curr_step > self.memory.batch_size:
            self.learn()
    
    def learn(self):
        for n, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = self.memory.sample()
            whole_next_actions = torch.stack([
                    ag.actor_target(ns) for ns, ag in
                    zip(next_states.swapaxes(0, 1), self.agents)]).swapaxes(0, 1).flatten(1)
            state = states[:, n]
            reward = rewards[:, n]
            next_state = next_states[:, n]
            done = dones[:, n]
            transitions = (state, actions, reward, next_state, done)
            agent.learn(transitions, states, next_states, whole_next_actions)
            agent.soft_update(agent.actor, agent.actor_target)
            agent.soft_update(agent.critic, agent.critic_target)
    
    def reset_noise(self):
        for agent in self.agents:
            agent.reset_noise()


class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu: np.ndarray = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
        self.state = self.mu.copy()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        self.state += self.theta * (self.mu - self.state) + self.sigma * self.rng.normal(
                size=len(self.state))
        return self.state


class GaussianNoise:
    def __init__(self, size, seed, mu=0., sigma=1.):
        self.mu = mu * np.ones(size)
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
        self.sigma = sigma
    
    def reset(self):
        pass
    
    def sample(self):
        return self.rng.normal(self.mu, self.sigma)


class ReplayBuffer:
    def __init__(self, batch_size: int, buffer_size: Union[int, float], num_agents: int, seed: int,
                 device: torch.device):
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.device = device
        
        self.states = None
        self.actions = None
        self.dones = np.zeros((self.buffer_size, num_agents), dtype=np.long)
        self.rewards = np.zeros((self.buffer_size, num_agents), dtype=np.float32)
        self.next_states = None
        
        self.memory_pos = 0
        self.buffer_length = 0
        
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
    
    def add(self, state, action, reward, next_state, done):
        if self.buffer_length == 0:
            self.states = np.zeros((self.buffer_size, *state.shape))
            self.next_states = np.zeros((self.buffer_size, *next_state.shape))
            self.actions = np.zeros((self.buffer_size, *action.shape))
        
        self.states[self.memory_pos] = state
        self.next_states[self.memory_pos] = next_state
        self.actions[self.memory_pos] = action
        self.dones[self.memory_pos] = done
        self.rewards[self.memory_pos] = reward
        
        self.memory_pos = (self.memory_pos + 1) % self.buffer_size
        self.buffer_length += 1 - int(self.buffer_length == self.buffer_size)
    
    def sample(self):
        samples_id = self.rng.integers(0, self.buffer_length, self.batch_size)
        
        states, actions, rewards, next_states, dones = self[samples_id]
        
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.uint8)
        
        return states, actions, rewards, next_states, dones
    
    def __getitem__(self, item):
        return self.states[item], self.actions[item], self.rewards[item], self.next_states[item], \
               self.dones[item]
    
    def __len__(self):
        return self.buffer_length
