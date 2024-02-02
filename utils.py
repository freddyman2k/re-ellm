import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from llm import LLMBaseClass
from datetime import datetime
import numpy as np
import pandas as pd

### Utility functions for setup, training and evaluation

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
    
    now = datetime.now()
    # Format as YYYY-MM-DD_HH
    date_time_str = now.strftime("%Y-%m-%d_%H")
    exp_run_str += date_time_str  
    return exp_run_str

class TextEmbedder:
    """Uses a pretrained SBERT model to embed text into a vector representation that can be used by a SB3 Policy network. 
    Caches embeddings for each input to avoid recomputing them.
    """
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2', max_cache_size=1000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = SentenceTransformer(model_name, device=device)
        self.max_cache_size = max_cache_size
        self.embeddings = {}
    
    def embed(self, text):
        if text in self.embeddings:
            return self.embeddings[text]
        else:
            embedding = self.model.encode(text)
            if len(self.embeddings) < self.max_cache_size:
                # Only add to cache if it is not full yet, otherwise always we will need to embed again
                # TODO: Would be nice to have a cache that automatically removes the oldest or least used entries instead
                self.embeddings[text] = embedding
            return embedding

class SaveCacheCallback(BaseCallback):
    def __init__(self, language_model: LLMBaseClass):
        super(SaveCacheCallback, self).__init__()
        self.language_model = language_model

    def _on_step(self) -> bool:
        # Save the cache
        #TODO: when saving cache, should we load the global cache first and update with new keys? 
        # (in case we run multiple instances of the same language model)
        self.language_model.save_cache()
        return True
    


### Plotting functions
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_rewards(rewards, checkpoint_dir, title='', time_steps=None, window_size=10):
    x = np.arange(len(rewards)) if time_steps is None else time_steps
    plt.plot(x, np.array(rewards), color='steelblue')
    #plt.plot(x, np.array(rewards), 'o', color='steelblue')
    # Calculate moving average
    moving_avg = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
    plt.plot(x, moving_avg, color = "darkred",linewidth=2.1)  
    plt.legend(['Mean eval reward', 'Moving average'], loc='upper left')
    plt.title(title)
    x_label ='Episodes'  if time_steps is None else 'Time steps'
    plt.xlabel(x_label)
    plt.ylabel('Reward')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(3,3))  # Add x 1e4 at the right end of the axis
    plt.savefig(checkpoint_dir + "mean_episode_eval_rewards.png")
    plt.show()

def plot_moving_averages(reward_time_step_dict, title='', checkpoint_dir='eval_logs/', window_size=10):
    for model_name, (rewards, time_steps) in reward_time_step_dict.items():
        # Calculate moving average
        moving_avg = pd.Series(rewards).rolling(window=window_size, min_periods=1).mean().values
        plt.plot(time_steps, moving_avg, linewidth=2.1, label=f'{model_name}')  

    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(3,3))  # Add x 1e4 at the right end of the axis
    plt.savefig(checkpoint_dir + "moving_avg_model_comparision.png")
    plt.show()




