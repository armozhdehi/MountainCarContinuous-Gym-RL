import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

from .ActorNet import ActorNet  # Adjusted for relative import
from .CriticNet import CriticNet  # Adjusted for relative import
from utils.Memory import ReplayBuffer  # Importing from utils
from utils.Noise import Noise  # Importing from utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent:
    """
    Agent that interacts with and learns from the environment using the TD3 algorithm.
    """
    def __init__(self, state_size, action_size, random_seed, batch_size, buffer_size, gamma, tau, actor_lr, critic_lr, weight_decay, noise_scale, policy_delay):
        """
        Initialize an Agent object.
        
        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            random_seed (int): Random seed for reproducibility.
            batch_size (int): Number of experiences to sample from memory.
            buffer_size (int): Maximum size of memory.
            gamma (float): Discount factor.
            tau (float): Soft update of target parameters.
            actor_lr (float): Learning rate for the actor network.
            critic_lr (float): Learning rate for the critic network.
            weight_decay (float): L2 weight decay.
            noise_scale (float): Scale of the noise added to the action.
            policy_delay (int): Delay for updating the policy.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.policy_delay = policy_delay

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNet(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNet(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        # Critic Network 1 (w/ Target Network)
        self.critic1_local = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic1_target = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # Critic Network 2 (w/ Target Network)
        self.critic2_local = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic2_target = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic2_optimizer = optim.Adam(self.critic2_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        # Noise process
        self.noise = Noise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

        # Initialize target networks
        self.update_targets(tau=1.0)

        self.learning_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        self.memory.add(state, action, reward, next_state, done)

        self.learning_step += 1
        if self.learning_step % self.policy_delay == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + Y * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Args:
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # Ensure all tensors are on the same device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)
        noise = torch.randn_like(next_actions) * self.noise_scale
        noise = torch.clamp(noise, -0.5, 0.5)
        next_actions = (next_actions + noise).clamp(-1, 1)

        Q_targets_next1 = self.critic1_target(next_states, next_actions)
        Q_targets_next2 = self.critic2_target(next_states, next_actions)
        Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected1 = self.critic1_local(states, actions)
        Q_expected2 = self.critic2_local(states, actions)
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets)
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets)

        # Minimize the loss
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        if self.learning_step % self.policy_delay == 0:
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic1_local(states, actions_pred).mean()

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1_local, self.critic1_target, self.tau)
            self.soft_update(self.critic2_local, self.critic2_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_targets(self, tau):
        """
        Update target networks with model parameters.
        """
        self.soft_update(self.actor_local, self.actor_target, tau)
        self.soft_update(self.critic1_local, self.critic1_target, tau)
        self.soft_update(self.critic2_local, self.critic2_target, tau)

    def save_checkpoint(self, path):
        """
        Save model parameters to a checkpoint file.
        """
        checkpoint = {
            "actor_local": self.actor_local.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_local": self.critic1_local.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_local": self.critic2_local.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        """
        Load model parameters from a checkpoint file.
        """
        checkpoint = torch.load(path)
        self.actor_local.load_state_dict(checkpoint["actor_local"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic1_local.load_state_dict(checkpoint["critic1_local"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_local.load_state_dict(checkpoint["critic2_local"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        print(f"Checkpoint loaded from {path}")