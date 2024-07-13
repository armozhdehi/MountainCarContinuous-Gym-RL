import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, policy, learning_rate, gamma):
        self.gamma = gamma
        self.policy = policy  # Ensure policy is already on the correct device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reward_memory = []
        self.log_prob_memory = []

    def store_transition(self, log_prob, reward):
        self.log_prob_memory.append(log_prob)
        self.reward_memory.append(reward)

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(self.policy.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if state.size(0) == 1:
            state = state.repeat(2, 1)  # Duplicate the state to make batch size 2
        state = state.to(self.policy.device)  # Ensure state is on the same device as the model
        action, log_prob = self.policy.act(state)
        return action[0], log_prob[0]

    def learn(self):
        self.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = torch.tensor(G, dtype=torch.float).to(device)
        policy_loss = []
        for g, log_prob in zip(G, self.log_prob_memory):
            policy_loss.append(-g * log_prob)
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.reward_memory = []
        self.log_prob_memory = []
