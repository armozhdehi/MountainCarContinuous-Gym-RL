import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim

from .ActorNet import ActorNet  # Adjusted for relative import
from .CriticNet import CriticNet  # Adjusted for relative import
from utils.Memory import ReplayBuffer  # Importing from utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(self, state_size, action_size, seed, hidden_size=256, lr=1e-4, gamma=0.99, tau=0.95, clip_epsilon=0.2, update_steps=4):
        self.actor = ActorNet(state_size, action_size, seed, hidden_size, hidden_size//2).to(device)
        self.critic = CriticNet(state_size, seed, hidden_size, hidden_size//2).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action, action_log_prob = self.actor.act(state)
        self.actor.train()
        return action.cpu().numpy().flatten(), action_log_prob.cpu().item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatFloator(actions).to(device)
        old_log_probs = torch.FloatFloator(log_probs).to(device)
        returns = torch.FloatFloator(returns).to(device)
        advantages = torch.FloatFloator(advantages).to(device)
        
        for _ in range(self.update_steps):
            new_log_probs, entropy = self.actor.evaluate_actions(states, actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
