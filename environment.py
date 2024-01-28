import gymnasium as gym
import text_crafter.text_crafter
from gymnasium.wrappers import LazyFrames
from collections import deque
import numpy as np
from gymnasium.spaces import Box  

class TransformObsSpace(gym.ObservationWrapper):
    # wrapp the observation space to include the embedded text observation and the image in a gym.spaces.Dict
    def __init__(self, env, embedding_shape=(384,)):
        super().__init__(env)
        self.observation_space=gym.spaces.Dict({
            'obs': self.env.observation_space,
            'text_obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=embedding_shape),
        })
    def observation(self, observation):
        text_obs_complete = " ".join([observation['text_obs'], observation['inv_status']['inv'], observation['inv_status']['status']])
        observation = {'obs': observation['obs'], 'text_obs': text_obs_complete}
        return observation

    
class CustomFrameStack(gym.ObservationWrapper):
    def __init__(self, 
                 env, 
                 num_stack, 
                 lz4_compress=False
    ):
        """Customized observation wrapper that stacks the image observations in a rolling manner.
        Adapted from FrameStack in gym.wrappers
        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        observation['obs'] = LazyFrames(list(self.frames), self.lz4_compress)
        return observation

    def step(self, action):
        """Steps through the environment, appending the image observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            dictionary of stacked imgs, and current observation of other items, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation['obs'])
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            dictionary of stacked imgs, and current observation of other items
        """
        observation, info = self.env.reset(**kwargs)
        [self.frames.append(observation['obs']) for _ in range(self.num_stack)]
        return self.observation(observation), info
    


