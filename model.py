import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hyperparams):
        super(QNetwork, self).__init__()
        layer1_size = hyperparams.get("q_fc1_units", 256)
        layer2_size = hyperparams.get("q_fc1_units", 256)
        init_w = hyperparams.get("q_init_w", 3e-3)
        self.linear1 = nn.Linear(state_size + action_size, layer1_size)
        self.linear2 = nn.Linear(layer1_size, layer2_size)
        self.linear3 = nn.Linear(layer2_size, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class GaussianPolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hyperparams):
        super(GaussianPolicyNetwork, self).__init__()
        layer1_size = hyperparams.get("p_fc1_units", 256)
        layer2_size = hyperparams.get("p_fc2_units", 256)
        init_w = hyperparams.get("p_init_w", 3e-3)
        self.log_std_min = hyperparams.get("p_log_std_min", -20)
        self.log_std_max = hyperparams.get("p_log_std_max", 2)

        self.linear1 = nn.Linear(state_size, layer1_size)
        self.linear2 = nn.Linear(layer1_size, layer2_size)

        # head for the mean
        self.mean_linear = nn.Linear(layer2_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        # head for the log(covariance)
        self.log_std_linear = nn.Linear(layer2_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        EPSILON=1e-6
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # to obtain actions, we sample a z-value from the obtained Gaussian distribution
        # later, we will take the hyperbolic tangent of the z value to obtain our action.
        # (see below in the post).
        z = normal.rsample()
        action = torch.tanh(z)

        # we modify the log_pi computation as explained in the Haarnoja et al. paper
        log_pi = (normal.log_prob(z) - torch.log(1 - (torch.tanh(z)).pow(2) + EPSILON)).sum(1, keepdim=True)

        return action, log_pi
