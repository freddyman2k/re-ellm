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
import yaml
import argparse


from llm import HuggingfacePipelineLLM, LLMGoalGenerator, ConstantGoalGenerator, ConstantSamplerGoalGenerator, TestCacheLLM
from ellm_reward import ELLMRewardCalculator
from utils import SaveCacheCallback, TextEmbedder, make_exp_name
from env_wrapper import CreateCompleteTextObs, RewardIfActionSimilarToGoalSuggestionsFromLastStep, EmbedTextObs, GenerateGoalSuggestions
import text_crafter.text_crafter


# TODO: Do we want this to be here or in the env_wrapper?
def make_env(name='CrafterTextEnv-v1',
                action_space_type='harder',
                device='cpu',  
                **kwargs):
    env = gym.make(name,
                action_space_type=action_space_type,
                device=device)
    return env

# TODO: Do we want this to be here or in the env_wrapper?
class SharedState:
    def __init__(self):
        self.last_state_goal_suggestions = None
        self.prev_achieved_goals = []


def make_full_train_env(env_spec, 
                        reward_calculator, 
                        goal_generator, 
                        obs_embedder, 
                        shared_state, device="cpu"):
    env = make_env(**env_spec, device=device)
    env = CreateCompleteTextObs(env)
    if not env_spec['env_reward']: #Test if agent learns from scratch
        env = RewardIfActionSimilarToGoalSuggestionsFromLastStep(env, reward_calculator, shared_state, similarity_threshold=env_spec['similarity_threshold'], check_ac_success=env_spec['check_ac_success'])
        env = GenerateGoalSuggestions(env, goal_generator, shared_state, novelty_bias=env_spec['novelty_bias']) 
    env = EmbedTextObs(env, obs_embedder)
    return env


def make_full_eval_env(env_spec, obs_embedder, device="cpu"):
    env = make_env(**env_spec, device=device)
    env = CreateCompleteTextObs(env)
    env = EmbedTextObs(env, obs_embedder)
    return env

def train_agent(env_spec: dict,
                 max_env_steps:int =5000000, 
                 eval_every:int=5000, 
                 log_every:int=1000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set up logging
    exp_run_str = make_exp_name(env_spec)
    eval_log_dir = f"./eval_logs/{exp_run_str}/"
    print("Logging eval results to", eval_log_dir)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Save env_spec to file
    with open(eval_log_dir + "env_spec.txt", "w") as f:
        f.write(str(env_spec))
    
    if env_spec['goal_generator'] == "LLMGoalGenerator":
        if env_spec['language_model'] == "mistral7binstruct":
            language_model = HuggingfacePipelineLLM("mistralai/Mistral-7B-Instruct-v0.2", cache_file="cache.pkl")
        elif env_spec['language_model'] == "testllm":
            language_model = TestCacheLLM() # for debugging other parts that do not need GPU, if you use this you don't need to submit a job to the cluster
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
    check_env(make_full_train_env(env_spec=env_spec,
                                  reward_calculator = reward_calculator, 
                                  goal_generator = goal_generator, 
                                  obs_embedder=obs_embedder, 
                                  shared_state=shared_state,
                                  ))
    check_env(make_full_eval_env(env_spec=env_spec,obs_embedder=obs_embedder))
    
    train_env = make_vec_env(make_full_train_env, 
                             n_envs=1, 
                             seed=env_spec['seed'], 
                             env_kwargs={'env_spec': env_spec,
                                         'reward_calculator': reward_calculator, 
                                         'goal_generator': goal_generator, 
                                         'obs_embedder': obs_embedder, 
                                         'shared_state': shared_state, 
                                         'device': device
                                         })
    train_env = VecFrameStack(train_env, n_stack=env_spec['frame_stack'])
    
    eval_env = make_vec_env(make_full_eval_env, 
                             n_envs=1, 
                             seed=env_spec['seed'], 
                             env_kwargs={'env_spec': env_spec,
                                        'obs_embedder': obs_embedder,
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
    parser = argparse.ArgumentParser(description='Train agent with config file')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Path to config file')
    parser.add_argument('--goal_generator', type=str, help='Valid options: LLMGoalGenerator, ConstantGoalGenerator, ConstantSamplerGoalGenerator', required=False)
    parser.add_argument('--novelty_bias', action='store_true', default=None, help='Enable novelty bias', required=False)
    parser.add_argument('--check_ac_success',action='store_true', default=None, help='Enable check if action was successful', required=False)
    
    args = parser.parse_args()
    print(args)

    with open(args.config, "r") as yamlfile:
        env_spec = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # add arguments from command line to env_spec
    if args.goal_generator:
        env_spec['goal_generator'] = args.goal_generator
    if args.novelty_bias:
        env_spec['novelty_bias'] = args.novelty_bias
    if args.check_ac_success:
        env_spec['check_ac_success'] = args.check_ac_success
    print(env_spec)
    train_agent(env_spec)