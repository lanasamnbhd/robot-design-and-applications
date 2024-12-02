import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from typing import Optional, Tuple



class Samples():
    def __init__(self, device: torch.device,
                 obs: np.ndarray, 
                 obs_next: np.ndarray, 
                 act: np.ndarray, 
                 rew: np.ndarray, 
                 done: np.ndarray):
        self.obs = torch.tensor(obs, device=device, dtype=torch.float32)
        self.obs_next = torch.tensor(obs_next, device=device, dtype=torch.float32)
        self.act = torch.tensor(act, device=device, dtype=torch.float32)
        self.rew = torch.tensor(rew, device=device, dtype=torch.float32)
        self.done = torch.tensor(done, device=device, dtype=torch.float32)


class ReplayBuffer():
    def __init__(self, env: gymnasium.Env,
                 buffer_size: int,
                 device: torch.device):
        self.buffer_size = buffer_size
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape

        self.ptr = 0
        self.full = False
        self.device = device

        self.obs_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=env.observation_space.dtype)
        self.obs_next_array = np.zeros((self.buffer_size, *self.obs_shape), dtype=env.observation_space.dtype)
        self.act_array = np.zeros((self.buffer_size, *self.act_shape), dtype=env.action_space.dtype)
        self.rew_array = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.done_array = np.zeros((self.buffer_size, 1), dtype=np.float32)


    def add(self, obs: np.ndarray, obs_next: np.ndarray, act: np.ndarray, rew: float, done: int):
        self.obs_array[self.ptr] = np.array(obs)
        self.obs_next_array[self.ptr] = np.array(obs_next)
        self.act_array[self.ptr] = np.array(act)
        self.rew_array[self.ptr] = np.array(rew)
        self.done_array[self.ptr] = np.array(done)

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
            self.ptr = 0
    

    def sample(self, batch_size: int) -> Samples:
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.ptr, size=batch_size)
        data = (self.index(self.obs_array, batch_inds),
                self.index(self.obs_next_array, batch_inds),
                self.index(self.act_array, batch_inds),
                self.index(self.rew_array, batch_inds),
                self.index(self.done_array, batch_inds))
        return Samples(self.device, *data)
    

    def index(self, x, inds):
        return x[inds] if x.ndim==1 else x[inds, :] 
