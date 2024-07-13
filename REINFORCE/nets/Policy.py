import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, state_size, action_size, seed=42, fc1_units=256, fc2_units=128):
        super(Policy, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units).to(self.device)
        self.bn1 = nn.BatchNorm1d(fc1_units).to(self.device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(self.device)
        self.bn2 = nn.BatchNorm1d(fc2_units).to(self.device)
        self.fc3 = nn.Linear(fc2_units, action_size).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, action_size)).to(self.device)
        self.reset_parameters()

    def forward(self, x, apply_bn=True):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.fc1(x))) if apply_bn else F.relu(self.fc1(x))
        x = F.relu(self.bn2(self.fc2(x))) if apply_bn else F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def act(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.to(device)  
        apply_bn = state.size(0) > 1  # Check if batch size is greater than 1
        mean, std = self.forward(state, apply_bn=apply_bn)
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(1, keepdim=True)
        return action.cpu().detach().numpy(), log_prob

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.log_std, -0.5)
