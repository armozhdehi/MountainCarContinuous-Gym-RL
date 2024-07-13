import numpy as np
import random
import copy

class Noise:
    """
    Ornstein-Uhlenbeck process for generating noise, often used to encourage exploration in the action space.
    """
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        
        Args:
            size (int): Dimension of the action space.
            seed (int): Random seed for reproducibility.
            mu (float): Long-running mean.
            theta (float): Speed of mean reversion.
            sigma (float): Volatility parameter.
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
