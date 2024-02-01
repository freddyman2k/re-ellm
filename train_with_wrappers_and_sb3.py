import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common import logger
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_checker import check_env
import numpy as np
import torch

from environment import CustomFrameStack, TransformObsSpace
from llm import HuggingfacePipelineLLM, LLMGoalGenerator, TestCacheLLM, ConstantGoalGenerator, ConstantSamplerGoalGenerator
from policy import DQNPolicy
from ellm_reward import ELLMRewardCalculator
from utils import SaveCacheCallback, TextEmbedder
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
    'check_ac_success': False,
    'novelty_bias': False,
    'goal_generator': "LLMGoalGenerator",
    'language_model': "mistral7binstruct",
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
    def __init__(self, env, reward_calculator, shared_state, similarity_threshold=0.99, check_ac_success=True, novelty_bias=True, print_on_reward=False):
        super().__init__(env)
        self.reward_calculator = reward_calculator
        self.similarity_threshold = similarity_threshold
        self.shared_state = shared_state
        self.print_on_reward = print_on_reward
        self.check_ac_success = check_ac_success
        self.novelty_bias = novelty_bias
        
        self.last_text_obs = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_text_obs = obs['text_obs']
        return obs, info
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
                
        # Compute reward based on similarity between the actual action and the suggested ones
        reward = 0
        action_name = self.env.get_action_name(action)
        intrinsic_reward, closest_suggestion = self.reward_calculator.compute_cosine_similarity(action_name, self.shared_state.last_state_goal_suggestions)
        if intrinsic_reward > self.similarity_threshold and ((info["action_success"] and self.check_ac_success) or not self.check_ac_success):
            reward = reward + intrinsic_reward
            if self.novelty_bias:
                self.shared_state.prev_achieved_goals.append(closest_suggestion)
            
            if self.print_on_reward:
                # Print some info if the agent received an intrinsic reward 
                print("\n=====================================================")
                print("\nRewarding agent for successfully enacting a suggestion!")
                print("\nLast text obs: ", self.last_text_obs)
                print("\nPolicy chose the following action from the last observation: ", action_name)
                print("\nGenerated suggestions in last step: ", self.shared_state.last_state_goal_suggestions)
                print("\nMost similar suggestion: ", closest_suggestion)
                print("\nCurrent text obs: ", obs['text_obs'])
                print("\nIntrinsic reward: ", intrinsic_reward)
                print("\nGoal suggestions after appending achieved suggestion: ", self.shared_state.prev_achieved_goals)
                print("=====================================================\n")
        
        self.last_text_obs = obs['text_obs']
        
        return obs, reward, terminated, truncated, info

class GenerateGoalSuggestions(gym.ObservationWrapper):
    """Generate goal suggestions for the agent to pursue, by prompting a language model with the current text observation"""
    def __init__(self, env, goal_generator, shared_state, novelty_bias=True):
        super().__init__(env)
        self.goal_generator = goal_generator
        self.shared_state = shared_state
        self.novelty_bias = novelty_bias
        
    def observation(self, obs):
        goal_suggestions = self.goal_generator.generate_goals(obs['text_obs'])
        if self.novelty_bias:
            # Remove goals that have already been achieved in previous steps
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
    env = RewardIfActionSimilarToGoalSuggestionsFromLastStep(env, reward_calculator, shared_state, similarity_threshold=SIMILARITY_THRESHOLD, check_ac_success=env_spec['check_ac_success'])
    env = GenerateGoalSuggestions(env, goal_generator, shared_state, novelty_bias=env_spec['novelty_bias']) 
    env = EmbedTextObs(env, obs_embedder)
    return env

def make_full_eval_env(obs_embedder, device="cpu"):
    env = make_env(**env_spec, device=device)
    env = CreateCompleteTextObs(env)
    env = EmbedTextObs(env, obs_embedder)
    return env

def make_exp_name(env_spec):
    exp_run_str = f"run_seed-{env_spec['seed']}_{env_spec['goal_generator']}_"
    if env_spec['goal_generator'] == "LLMGoalGenerator":
        exp_run_str += f"{env_spec['language_model']}_"
    if env_spec['novelty_bias']:
        exp_run_str += "novelty-bias_"
    else:
        exp_run_str += "no-novelty-bias_"
    if env_spec['check_ac_success']:
        exp_run_str += "reward-on-ac-success_"
    else:
        exp_run_str += "ignore-ac-success_"
        
    return exp_run_str
    

def train_agent(max_env_steps=5000000, eval_every=5000, log_every=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up logging
    exp_run_str = make_exp_name(env_spec)
    eval_log_dir = f"./eval_logs/{exp_run_str}/"
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Save env_spec to file
    with open(eval_log_dir + "env_spec.txt", "w") as f:
        f.write(str(env_spec))
    
    if env_spec['goal_generator'] == "LLMGoalGenerator":
        # language_model = TestCacheLLM() # for debugging other parts that do not need GPU, if you use this you don't need to submit a job to the cluster
        if env_spec['language_model'] == "mistral7binstruct":
            language_model = HuggingfacePipelineLLM("mistralai/Mistral-7B-Instruct-v0.2", cache_file="cache.pkl")
        else:
            raise ValueError("Unknown language model, needs to be implemented first maybe?")
        goal_generator = LLMGoalGenerator(language_model=language_model)
    elif env_spec['goal_generator'] == "ConstantGoalGenerator":
        env_for_names = make_env(**env_spec)
        goal_generator = ConstantGoalGenerator(goal_list=[env_for_names.get_action_name(i) for i in range(env_for_names.action_space.n)])
    elif env_spec['goal_generator'] == "ConstantSamplerGoalGenerator":
        env_for_names = make_env(**env_spec)
        goal_generator = ConstantSamplerGoalGenerator(goal_list=[env_for_names.get_action_name(i) for i in range(env_for_names.action_space.n)])
        
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
    callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir, eval_freq=eval_every,
                                n_eval_episodes=10, deterministic=True,
                                render=False)
    
    if env_spec['goal_generator'] == "LLMGoalGenerator":
        # If we use an LLMGoalGenerator, we need to save the cache every eval_every steps
        save_cache_callback = EveryNTimesteps(n_steps=eval_every, callback=SaveCacheCallback(language_model))
        callback = [callback, save_cache_callback]
    
    tensorboard_log_dir = f"./tb_logs/{exp_run_str}/"
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    agent = DQN('MultiInputPolicy', 
                train_env, 
                verbose=1, 
                tensorboard_log=tensorboard_log_dir, 
                device=device)
    
    agent.learn(total_timesteps=max_env_steps, 
                callback=callback, 
                log_interval=log_every)
    
if __name__ == "__main__":    
    train_agent()