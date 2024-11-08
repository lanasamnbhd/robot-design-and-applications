import gymnasium
from torch.utils.tensorboard import SummaryWriter
 
from ppo.ppo import PPO
from env.wrapper import ImageEnv
 
env = gymnasium.make("CarRacing-v2")
env = ImageEnv(env)
 
agent = PPO(env, ckpt_path='saved/ppo_ImageEnv__06_18_2024_08_43/505.pt', epoch_steps=4000)
 
agent.learn()
