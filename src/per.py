from typing import Union

import numpy as np
import torch

from sum_tree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, batch_size: int, buffer_size: Union[int, float], beta, alpha: float,
                 num_agents: int, seed: int, device: torch.device):
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.device = device
        
        self.states = None
        self.actions = None
        self.dones = np.zeros((self.buffer_size, num_agents), dtype=np.long)
        self.rewards = np.zeros((self.buffer_size, num_agents), dtype=np.float32)
        self.next_states = None
        
        self.sum_tree = SumTree(np.zeros((self.buffer_size,)), seed=seed)
        
        self.memory_pos = 0
        self.buffer_length = 0
        
        self.alpha = alpha
        self.beta = beta
        
        self.rng = np.random.Generator(np.random.PCG64(seed=seed))
    
    def add(self, state, action, reward, next_state, done):
        if self.buffer_length == 0:
            self.states = np.zeros((self.buffer_size, *state.shape))
            self.next_states = np.zeros((self.buffer_size, *next_state.shape))
            self.actions = np.zeros((self.buffer_size, *action.shape))
            priority = 1
        else:
            priority = self.sum_tree.max
        
        self.states[self.memory_pos] = state
        self.next_states[self.memory_pos] = next_state
        self.actions[self.memory_pos] = action
        self.dones[self.memory_pos] = done
        self.rewards[self.memory_pos] = reward
        
        self.sum_tree.update(priority, self.memory_pos)
        
        self.memory_pos = (self.memory_pos + 1) % self.buffer_size
        self.buffer_length += 1 - int(self.buffer_length == self.buffer_size)
    
    def sample(self):
        nodes, samples_id = self.sum_tree.sample(self.batch_size)
        
        # samples_id = torch.multinomial(self.probabilities, self.batch_size)
        states, actions, rewards, next_states, dones = self[samples_id]
        
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.uint8)
        
        return states, actions, rewards, next_states, dones, nodes, samples_id
    
    def calc_is_weight(self, nodes_value):
        """
        Calculate Importance-Sampling weights for bias correction
        :param nodes_value: list of nodes for each sampled experience in this time step
        :return: IS weights
        """
        beta = self.beta.step()
        nodes_value = torch.tensor(nodes_value)
        sample_probabilities = nodes_value / self.sum_tree.top_node.value
        weights = ((1 / (len(self) * sample_probabilities.to(self.device))) ** beta)
        weights /= weights.max()
        return weights
    
    def update(self, loss, samples_id) -> None:
        """
        Update priorities for every experience

        :param loss: TD-difference from the last gradient descent step
        :param samples_id: list of IDs for each sampled experience in this time step
        """
        self.sum_tree.bulk_update(torch.abs(loss + 1e-6).pow(self.alpha).cpu(), samples_id)
    
    def __getitem__(self, item):
        return self.states[item], self.actions[item], self.rewards[item], self.next_states[item], \
               self.dones[item]
    
    def __len__(self):
        return self.buffer_length


class Annealing:
    def __init__(self, start, end, steps):
        self.step_ = (end - start) / steps
        self.value = start
        self.end = end
        self.start = start
    
    def step(self):
        self.value += self.step_
        return self.value
