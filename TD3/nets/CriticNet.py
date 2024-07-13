import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialization import param_init

class CriticNet(nn.Module):
    """Critic (Value) Model for evaluating the value of state-action pairs."""
    
    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """
        Initialize parameters and build the Critic model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed for reproducibility
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using uniform distribution based on fan-in of the layer."""
        self.fcs1.weight.data.uniform_(*param_init(self.fcs1))
        self.fc2.weight.data.uniform_(*param_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Defines the forward pass of the critic model.
        Args:
            state (Tensor): The input state.
            action (Tensor): The action taken.
        Returns:
            Tensor: The Q-value as predicted by the network.
        """
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)