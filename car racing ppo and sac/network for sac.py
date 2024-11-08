import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F



LOG_STD_MAX = 2
LOG_STD_MIN = -5



def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



class Encoder(nn.Module):
    def __init__(self, in_shape, out_size) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(in_shape[0], 16, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=4, stride=2)),
            nn.ReLU(),
            nn.Flatten()
        )

        # compute conv output size
        with torch.inference_mode():
            output_size = self.conv(torch.zeros(1, *in_shape)).shape[1]
        self.fc = layer_init(nn.Linear(output_size, out_size))

    def forward(self, x):
        x = self.conv(x/255.0)
        x = self.fc(x)
        return x



class CNNQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape

        self.obs_encoder = Encoder(obs_shape, 256)
        self.act_encoder = nn.Linear(act_shape[0], 256)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, obs, act):
        obs_encoding = F.relu(self.obs_encoder(obs))
        act_encoding = F.relu(self.act_encoder(act))
        x = torch.cat([obs_encoding, act_encoding], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class CNNActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        if obs_shape[0] == obs_shape[1]:
            h, w, c = obs_shape
            obs_shape = (c, h, w)

        self.obs_encoder = Encoder(obs_shape, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_shape[0])
        self.fc_log_std = nn.Linear(256, act_shape[0])

        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    
    def forward(self, obs):
        obs_encoding = F.relu(self.obs_encoder(obs))
        x = F.relu(self.fc1(obs_encoding))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
    

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()

        normal = Normal(mean, std)
        # for reparameterization trick (mean + std * N(0,1))
        # this allows backprob with respect to mean and std
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
