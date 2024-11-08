import torch
import scipy
import numpy as np
import gymnasium
 
 
 
def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute the discounted cumulative sum
 
    Input:
        x = [x0, x1, x2]
    Output:
        [x0 + d * x1 + d^2 * x2, x1 + d * x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
 
 
 
def gae_lambda(rews: np.ndarray, vals: np.ndarray, gamma: float, lam: float=0.97) -> np.ndarray:
    deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    return discount_cumsum(deltas, gamma*lam)
 
 
 
class Samples():
    def __init__(self, device: torch.device,
                 obs: np.ndarray,
                 act: np.ndarray,
                 logp: np.ndarray,
                 rtg: np.ndarray,
                 adv: np.ndarray):
        self.obs = torch.tensor(obs, device=device, dtype=torch.float32)
        self.act = torch.tensor(act, device=device, dtype=torch.float32)
        self.logp = torch.tensor(logp, device=device, dtype=torch.float32)
        self.rtg = torch.tensor(rtg, device=device, dtype=torch.float32)
        self.adv = torch.tensor(adv, device=device, dtype=torch.float32)
 
 
 
class PPOBuffer:
    def __init__(self, env: gymnasium.Env, buffer_size: int, device: torch.device, gamma: float=0.99, lamb: float=0.97) -> None:
        self.buffer_size = buffer_size
        self.obs_shape = env.observation_space.shape
        self.act_shape = env.action_space.shape
        self.device = device
 
        self.gamma = gamma
        self.lamb = lamb
       
        self.obs_buf = np.zeros((self.buffer_size, *self.obs_shape), dtype=np.float32)
        self.act_buf = np.zeros((self.buffer_size, *self.act_shape), dtype=np.float32)
        self.rew_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.val_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.logp_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.rtg_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.adv_buf = np.zeros((self.buffer_size, 1), dtype=np.float32)
 
        self.ptr, self.path_start_idx = 0, 0
   
 
    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float) -> None:
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
 
 
    def path_done(self, last_val: float) -> None:
        """
        Call this after each episode to compute reward-to-gos and advantages.
 
        Input:
            - last-val (float): 0 if the trajectory ended because the agent
              reached a terminal state (died), and V(s_T) otherwise (cut-off).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val) # shape = (N,)
        vals = np.append(self.val_buf[path_slice], last_val)
 
        self.rtg_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1].reshape(-1, 1)
        self.adv_buf[path_slice] = gae_lambda(rews, vals, self.gamma).reshape(-1, 1)
 
        self.path_start_idx = self.ptr
       
 
    def get(self) -> Samples:
        data = Samples(self.device, self.obs_buf, self.act_buf, self.logp_buf, self.rtg_buf, self.adv_buf)
        self.ptr, self.path_start_idx = 0, 0
        return data
