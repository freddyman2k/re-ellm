import random
from buffer import ReplayBuffer

class DQNPolicy:
    def __init__(self, observation_shapes, n_actions):
        self.model = None
        self.observation_shapes = observation_shapes
        self.n_actions = n_actions
        self.buffer = ReplayBuffer(1000, observation_shapes, n_actions)

    def select_action(self, state):
        # TODO: Implement this. Returns dummy random action for now
        return random.randint(0, self.n_actions - 1)
    
    def update(self, batch_size=64):
        # Sample batch from replay buffer and update model
        pass
