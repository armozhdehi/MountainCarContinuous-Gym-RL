import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialization import param_init

class CriticNet(nn.Module):
    """Defines a Critic (Value) Model for evaluating state values."""

    def __init__(self, state_size, seed, fc1_units=256, fc2_units=128):
        """
        Initializes the Critic model with two hidden layers.
        Args:
            state_size (int): Dimension of each state
            seed (int): Random seed for reproducibility
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(CriticNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using uniform distribution based on fan in of the layer."""
        self.fc1.weight.data.uniform_(*param_init(self.fc1))
        self.fc2.weight.data.uniform_(*param_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Defines the forward pass of the critic model.
        Args:
            state (Tensor): The input state.
        Returns:
            Tensor: The state value.
        """
        x = self.fc1(state)
        if x.size(0) > 1:  # Apply BatchNorm only if batch size > 1
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:  # Apply BatchNorm only if batch size > 1
            x = self.bn2(x)
        x = F.relu(x)
        
        return self.fc3(x)
