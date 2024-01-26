import gym
import text_crafter.text_crafter
from gym.wrappers import LazyFrames
from collections import deque
import numpy as np
from gym.spaces import Box


env_spec = {
    'name': 'CrafterTextEnv-v1',
    'seed': 111,
    'action_space_type': 'harder',
    'env_reward': None,  # to be specified later
    'dying': True,
    'length': 400,
    'max_seq_len': 200,
    'use_sbert': False,
    'device': 'cpu',
    'use_language_state': False,
    'threshold': .99,
    'check_ac_success': True,
    'frame_stack': 4,
}

class TextCrafterEnv:
    def __init__(self, 
                name='CrafterTextEnv-v1',
                action_space_type='harder',
                env_reward=None,
                device='cpu',  
                use_language_state=False,
                use_sbert=False,
                frame_stack=4,
                **kwargs
                ):
        
        env = gym.make(name,
                action_space_type=action_space_type,
                env_reward=env_reward,
                device=device,  
                use_language_state=use_language_state,
                use_sbert=use_sbert,) 
        self.env = CustomFrameStack(env, frame_stack) 
        

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
    
    
    
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
    


env = TextCrafterEnv(**env_spec)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)

