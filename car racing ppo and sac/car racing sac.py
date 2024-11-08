import gymnasium
from torch.utils.tensorboard import SummaryWriter

from sac.sac import SAC
from env.wrapper import ImageEnv

env = gymnasium.make("CarRacing-v2")
env = ImageEnv(env)

agent = SAC(env, ckpt_path='saved/sac_ImageEnv__06_15_2024_22_52/569999.pt', alpha=0.2, batch_size=512)

agent.learn()
