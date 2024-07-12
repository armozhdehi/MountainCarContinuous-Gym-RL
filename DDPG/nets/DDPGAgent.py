import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

from .ActorNet import ActorNet  # Adjusted for relative import
from .CriticNet import CriticNet  # Adjusted for relative import
from utils.Noise import Noise  # Importing from utils
from utils.Memory import ReplayBuffer  # Importing from utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    """
    Agent that interacts with and learns from the environment using DDPG algorithm.
    """
    def __init__(self, state_size, action_size, random_seed, batch_size, buffer_size, gamma, tau, actor_lr, critic_lr, weight_decay):
        """
        Initializes an Agent with Actor and Critic networks, noise process, and replay buffer.
        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            random_seed (int): Random seed for reproducibility.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay

        self.seed = random.seed(random_seed)

        self.actor_local = ActorNet(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNet(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)

        self.critic_local = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=weight_decay)

        self.noise = Noise(action_size, random_seed)
        self.memory = ReplayBuffer(action_size, int(buffer_size), int(batch_size), random_seed)


    def step(self, state, action, reward, next_state, done):
        """
        Process a step received from the environment, save it to the replay buffer, and potentially
        initiate learning from a random sample of experiences if enough are available.

        Args:
            state (np.array): The current state of the environment.
            action (np.array): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (np.array): The next state of the environment after taking the action.
            done (bool): Boolean flag indicating if the episode has terminated.

        Returns:
            None
        """
        # Validate input types to ensure consistency
        assert isinstance(state, np.ndarray), "Expected state to be np.ndarray"
        assert isinstance(action, np.ndarray), "Expected action to be np.ndarray"
        assert isinstance(reward, (int, float)), "Expected reward to be int or float"
        assert isinstance(next_state, np.ndarray), "Expected next_state to be np.ndarray"
        assert isinstance(done, bool), "Expected done to be bool"

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Check if enough samples are available in the memory for learning
        if len(self.memory) >= self.memory.batch_size:
            experiences = self.memory.sample()
            self.train(experiences)


    def act(self, state, add_noise=True):
        """
        Returns the action to be taken given a state, according to the current policy.
        
        Args:
            state (np.array): The current state from the environment.
            add_noise (bool, optional): Flag to determine whether to add noise for exploration. Defaults to True.
        
        Returns:
            np.array: The action values clipped between -1 and 1.
        
        Raises:
            ValueError: If the input state is not in the expected format or dimension.
        """
        # Check if the input state is a NumPy array and has the correct dimension
        if not isinstance(state, np.ndarray) or state.ndim != 1:
            raise ValueError("Input state must be a one-dimensional NumPy array.")

        # Convert state to tensor and move it to the appropriate device
        state_tensor = torch.from_numpy(state).float().to(device)

        # Set the local actor model to evaluation mode
        self.actor_local.eval()

        # Disable gradient calculation
        with torch.no_grad():
            action = self.actor_local(state_tensor).cpu().data.numpy()

        # Set the local actor model back to training mode
        self.actor_local.train()

        # Optionally add noise to the action for exploration purposes
        if add_noise:
            action += self.noise.sample()

        # Clip the action values to ensure they are within the valid action space bounds
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()


    def train(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Calculates the Q targets, updates the Critic by minimizing the loss between Q targets and Q expected,
        and updates the Actor by using the policy's gradient.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples.

        Returns:
            None
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Freeze the critic network to prevent it from accumulating gradients during actor update
        for param in self.critic_local.parameters():
            param.requires_grad = False

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze the critic network after updating the actor
        for param in self.critic_local.parameters():
            param.requires_grad = True

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float):
        """
        Perform a soft update on the target model's parameters using the local model's parameters.
        It blends the parameters of both models according to the formula:
        
            θ_target = τ * θ_local + (1 - τ) * θ_target

        This method slowly blends the weights, helping stabilize learning in the target network.

        Args:
            local_model (torch.nn.Module): The local model from which weights will be copied.
            target_model (torch.nn.Module): The target model to which weights will be copied.
            tau (float): The interpolation parameter controlling the blending ratio.
        
        Returns:
            None
        """
        # Ensure gradient computations are not accidentally included in the graph
        with torch.no_grad():
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                if target_param.data.shape != local_param.data.shape:
                    raise ValueError("Parameter shapes between the local and target model do not match.")
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)