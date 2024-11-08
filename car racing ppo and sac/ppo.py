import torch
import numpy as np
import gymnasium
from env.wrapper import ImageEnv
from typing import Union, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm
from datetime import datetime
import os
 
from ppo.buffer import PPOBuffer, Samples
from ppo.network import CNNPolicy, CNNValue
       
 
 
class PPO:
    def __init__(self, env, **kwargs):
        self.env = env
        self._init_hyperparameters(**kwargs)
        self._init_seed()
        self._init_networks()
        self.buffer = PPOBuffer(env, buffer_size=self.epoch_steps, device=self.device, gamma=self.gam, lamb=self.lamb)
        # Logging
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPO_'+self.env_name+self.time_str)
        self.step, self.num_eps = 0, 0
 
 
    def learn(self):
        num_epochs = self.total_steps//self.epoch_steps
        for epoch in tqdm(range(num_epochs)):
            data = self.rollout()
            self.update(data, epoch)
 
            if (epoch+1)%self.save_freq == 0:
                self.save_ckpt(ckpt_name=f'{epoch}.pt')
 
            if (epoch+1)%self.val_freq == 0:
                self.validate(epoch)
 
 
    def rollout(self) -> Samples:
        eps_ret, eps_len = 0, 0
        obs, info = self.env.reset()
        for t in range(self.epoch_steps):
            act, logp, val = self.interact(obs)
            act_applied = np.clip(act, self.env.action_space.low, self.env.action_space.high)
            assert self.env.action_space.contains(act_applied), f"Invalid action {act_applied} in the action space"
            obs_next, rew, term, trun, info =  self.env.step(act_applied)
 
            self.buffer.add(obs, act, rew, val, logp)
            self.step += 1
            eps_ret, eps_len = eps_ret + rew, eps_len + 1
 
            obs = obs_next
 
            done = term or trun
            epoch_ended = (t==(self.epoch_steps-1))
            if done or epoch_ended:
                if trun or epoch_ended:
                    last_val = self.value(torch.Tensor(obs).unsqueeze(0).to(self.device)).detach().squeeze(0).cpu().numpy()[0]
                else:
                    last_val = 0
               
                # Only record the episode when it's not cut off by epoch size
                if not epoch_ended:
                    self.num_eps += 1
                    if self.use_tb:
                        self.writer.add_scalar('charts/episode_return', eps_ret, self.num_eps)
                        self.writer.add_scalar('charts/episode_length', eps_len, self.num_eps)
               
                self.buffer.path_done(last_val)
                # reset environment
                eps_ret, eps_len = 0, 0
                obs, info = self.env.reset()
 
        return self.buffer.get()
   
 
    def update(self, data: Samples, epoch: int):
        adv = data.adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-10) # advantage normalization
        # POLICY UPDATE
        for i_p in range(self.num_updates):
            logp_new = self.policy(data.obs, data.act)
            ratio = torch.exp(logp_new - data.logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_policy = (-torch.min(surr1, surr2)).mean()
 
            approx_kl = (data.logp - logp_new).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break
 
            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()
       
        # VALUE UPDATE
        for i_v in range(self.num_updates):
            # update value
            loss_value = ((self.value(data.obs) - data.rtg)**2).mean()
 
            self.value_optimizer.zero_grad()
            loss_value.backward()
            self.value_optimizer.step()
 
        if self.use_tb:
            self.writer.add_scalar('charts/num_policy_updates', i_p, epoch)
            self.writer.add_scalar('charts/num_value_updates', i_v, epoch)
            self.writer.add_scalar('charts/loss_policy', -loss_policy.item(), epoch)
            self.writer.add_scalar('charts/loss_value', loss_value.item(), epoch)
   
 
    def interact(self, obs: np.ndarray) -> Union[np.ndarray, float, float]:
        obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            act, logp, _ = self.policy(obs)
            val = self.value(obs)
        act = act.squeeze(0).cpu().numpy()
        logp = logp.squeeze(0).cpu().numpy()[0]
        val = val.squeeze(0).cpu().numpy()[0]
        return act, logp, val
   
 
    def validate(self, epoch):
        if self.render_val:
            env_val = gymnasium.make(self.env.spec.id, render_mode='human')
        else:
            env_val = gymnasium.make(self.env.spec.id)
        env_val = ImageEnv(env_val)
        obs, _ = env_val.reset()
        done = False
        eps_ret, eps_len = 0, 0
        while not done:
            with torch.no_grad():
                _, _, act = self.policy(torch.Tensor(obs).unsqueeze(0).to(self.device))
            act = act.squeeze(0).cpu().numpy()
            act = np.clip(act, self.env.action_space.low, self.env.action_space.high)
            obs, rew, term, trun, _ = env_val.step(act)
            done = term or trun
            eps_ret += rew
            eps_len += 1
 
        env_val.close()
        print(f"Validation Episode Return: {eps_ret}, Length: {eps_len}")
        if self.use_tb:
            self.writer.add_scalar('validation/episode_return', eps_ret, epoch)
            self.writer.add_scalar('validation/episode_length', eps_len, epoch)
 
 
    def _init_hyperparameters(self, **kwargs):
        self.env_name = self.env.__class__.__name__
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {self.device}')
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
 
        self.seed = kwargs.get('seed', 0)
        self.use_tb = kwargs.get('use_tb', True)
        self.ckpt_path = kwargs.get('ckpt_path', None)
 
        self.epoch_steps = kwargs.get('epoch_steps', 238*3)
        self.total_steps = kwargs.get('total_steps', int(3e6))
        self.clip_ratio = kwargs.get('clip_ratio', 0.2)
        self.num_updates = kwargs.get('num_updates', 80)
        self.gam = kwargs.get('gamma', 0.99)
        self.lamb = kwargs.get('lambda', 0.97)
        self.policy_lr = kwargs.get('policy_lr', 1e-4)
        self.value_lr = kwargs.get('v_lr', 1e-4)
        self.target_kl = kwargs.get('target_kl', 0.01)
        self.save_freq = kwargs.get('save_freq', 10000//self.epoch_steps)
        self.val_freq = kwargs.get('val_freq', 4000//self.epoch_steps)
        self.render_val = kwargs.get('render_val', True)
   
 
    def _init_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
 
 
    def _init_networks(self):
        self.policy = CNNPolicy(self.env).to(self.device)
        self.value = CNNValue(self.env).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=self.value_lr)
        if self.ckpt_path != None:
            self._load_ckpt(torch.load(self.ckpt_path))
 
 
    def save_ckpt(self, ckpt_name: str):
        directory = os.path.join('saved', f'ppo_{self.env_name}_{self.time_str}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, ckpt_name)
       
        torch.save({"policy_state_dict": self.policy.state_dict(),
                    "value_state_dict": self.value.state_dict()
                    }, path)
 
        print(f"Checkpoint saved to {path}")
 
 
    def _load_ckpt(self, ckpt: dict):
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value.load_state_dict(ckpt["value_state_dict"])
      
