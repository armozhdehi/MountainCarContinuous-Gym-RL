class ReplayBuffer:
    def __init__(self):
        self.clear()

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        if action is not None:
            self.actions.append(action)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if reward is not None:
            self.rewards.append(reward)
        if value is not None:
            self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []