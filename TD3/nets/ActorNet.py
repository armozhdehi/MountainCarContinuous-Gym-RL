import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.initialization import param_init

class ActorNet(nn.Module):
    """Defines an Actor (Policy) Model for a reinforcement learning environment."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        Initializes the Actor model with two hidden layers.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed for reproducibility
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(ActorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using uniform distribution based on fan in of the layer."""
        self.fc1.weight.data.uniform_(*param_init(self.fc1))
        self.fc2.weight.data.uniform_(*param_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Defines the forward pass of the actor model.
        Args:
            state (Tensor): The input state.
        Returns:
            Tensor: Output action values from the network.
        """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))