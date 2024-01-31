import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_checker import check_env
import numpy as np
import torch

from environment import CustomFrameStack, TransformObsSpace
from llm import HuggingfacePipelineLLM, LLMGoalGenerator, DummyLLM
from policy import DQNPolicy
from ellm_reward import ELLMRewardCalculator
from utils import TextEmbedder
import text_crafter.text_crafter


SIMILARITY_THRESHOLD = 0.99
BATCH_SIZE = 64
env_spec = {
    'name': 'CrafterTextEnv-v1',
    'action_space_type': 'harder',
    'env_reward': False, 
    'seed': 1,
    'dying': True,
    'length': 400, # TODO: Discuss if we should use a different length for training
    'threshold': .99,
    'check_ac_success': True,
    'frame_stack': 4,
}

def make_env(name='CrafterTextEnv-v1',
                action_space_type='harder',
                device='cpu',  
                **kwargs):
    env = gym.make(name,
                action_space_type=action_space_type,
                device=device)
    return env

class SharedState:
    def __init__(self):
        self.last_state_goal_suggestions = None
        self.prev_achieved_goals = []

class CreateCompleteTextObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    """Create the complete text observation as in the paper by concatenating the text observation with the inventory and status"""
    def observation(self, observation):
        text_obs_complete = " ".join([observation['text_obs'], observation['inv_status']['inv'], observation['inv_status']['status']])
        observation = {'obs': observation['obs'], 'text_obs': text_obs_complete}
        return observation
    
class RewardIfActionSimilarToGoalSuggestionsFromLastStep(gym.Wrapper):
    """Rewards the agent if the action it took is similar to one of the goal suggestions for the next state"""
    def __init__(self, env, reward_calculator, shared_state, similarity_threshold=0.99):
        super().__init__(env)
        self.reward_calculator = reward_calculator
        self.similarity_threshold = similarity_threshold
        self.shared_state = shared_state
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
                
        # Compute reward based on similarity between the actual action and the suggested ones
        action_name = self.env.get_action_name(action)
        intrinsic_reward, closest_suggestion = self.reward_calculator.compute_cosine_similarity(action_name, self.shared_state.last_state_goal_suggestions)
        if intrinsic_reward > SIMILARITY_THRESHOLD and info["action_success"]:
            reward = reward + intrinsic_reward
            self.shared_state.prev_achieved_goals.append(closest_suggestion)
        
        return obs, reward, terminated, truncated, info

class GenerateGoalSuggestions(gym.ObservationWrapper):
    """Generate goal suggestions for the agent to pursue, by prompting a language model with the current text observation"""
    def __init__(self, env, goal_generator, shared_state):
        super().__init__(env)
        self.goal_generator = goal_generator
        self.shared_state = shared_state
        
    def observation(self, obs):
        goal_suggestions = self.goal_generator.generate_goals(obs['text_obs'])
        goal_suggestions = [goal for goal in goal_suggestions if goal not in self.shared_state.prev_achieved_goals]
        self.shared_state.last_state_goal_suggestions = goal_suggestions # Save for reward calculation in next step, after policy has acted
        # TODO: Add goal suggestions to observation here, so that a goal-conditioned agent can be trained
        return obs
    
class EmbedTextObs(gym.ObservationWrapper):
    """Embed the text observation using the text embedder. Necessary for stable baselines to work with the text observation."""
    def __init__(self, env, obs_embedder):
        super().__init__(env)
        self.obs_embedder = obs_embedder
        
        # This is a little hacky, but the original text crafter env has a observation space that does not correspond to the actual output of the env
        self.observation_space = gym.spaces.Dict({
            'obs': self.env.observation_space,
            'text_obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_embedder.embed("test").shape),
        })

    def observation(self, observation):
        return {
            'obs': observation['obs'], 
            'text_obs': self.obs_embedder.embed(observation['text_obs'])
            }

def make_full_train_env(reward_calculator, goal_generator, obs_embedder, shared_state, device="cpu"):
    env = make_env(**env_spec, device=device)
    env = CreateCompleteTextObs(env)
    env = RewardIfActionSimilarToGoalSuggestionsFromLastStep(env, reward_calculator, shared_state, similarity_threshold=SIMILARITY_THRESHOLD)
    env = GenerateGoalSuggestions(env, goal_generator, shared_state) 
    env = EmbedTextObs(env, obs_embedder)
    return env

def make_full_eval_env(obs_embedder, device="cpu", ):
    env = make_env(**env_spec, device=device)
    env = CreateCompleteTextObs(env)
    env = EmbedTextObs(env, obs_embedder)
    return env

def train_agent(max_env_steps=5000000, eval_every=5000, log_every=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create log dir where evaluation results will be saved
    eval_log_dir = "./eval_logs/"
    os.makedirs(eval_log_dir, exist_ok=True)
    
    language_model = HuggingfacePipelineLLM("mistralai/Mistral-7B-Instruct-v0.2", cache_file="cache.pkl")
    # language_model = TestCacheLLM() # for debugging other parts that do not need GPU, if you use this you don't need to submit a job to the cluster
    goal_generator = LLMGoalGenerator(language_model=language_model)
    obs_embedder = TextEmbedder()
    reward_calculator = ELLMRewardCalculator()
    
    # Some things need to be shared between wrappers, so we use a shared state object
    shared_state = SharedState()
    
    # TODO: Discuss if maybe we should only use one ELLMWrapper which does everything (or at least Reward + Goal Generation, since this is where the global magic happens), instead of splitting it up into multiple wrappers -> could avoid shared state
    
    # Make sure that envs are valid. TODO: maybe move to tests
    check_env(make_full_train_env(reward_calculator, goal_generator, obs_embedder, shared_state))
    check_env(make_full_eval_env(obs_embedder))
    
    train_env = make_vec_env(make_full_train_env, 
                             n_envs=1, 
                             seed=env_spec['seed'], 
                             env_kwargs={'reward_calculator': reward_calculator, 
                                         'goal_generator': goal_generator, 
                                         'obs_embedder': obs_embedder, 
                                         'shared_state': shared_state, 
                                         'device': device
                                         })
    train_env = VecFrameStack(train_env, n_stack=env_spec['frame_stack'])
    
    eval_env = make_vec_env(make_full_eval_env, 
                             n_envs=1, 
                             seed=env_spec['seed'], 
                             env_kwargs={'obs_embedder': obs_embedder,
                                         'device': device})
    eval_env = VecFrameStack(eval_env, n_stack=env_spec['frame_stack'])

    # Create callback that evaluates agent every eval_every steps and saves the best model
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir, eval_freq=eval_every,
                                n_eval_episodes=10, deterministic=True,
                                render=False)
    
    agent = DQN('MultiInputPolicy', train_env, verbose=1, tensorboard_log="./tb_logs", device=device)
    agent.learn(total_timesteps=max_env_steps, callback=eval_callback, log_interval=log_every)
    
if __name__ == "__main__":    
    train_agent()