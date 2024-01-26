import gym

class TextCrafterEnv:
    def __init__(self):
        self.env = gym.make('TextCrafter')  

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
