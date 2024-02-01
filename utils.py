import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from llm import LLMBaseClass
from datetime import datetime


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
        
def visualize_obs(obs, goal_suggestions=None, frame_stack=4):
    fig, axs = plt.subplots(1, frame_stack, figsize=(5 * frame_stack, 5))
    title = obs["text_obs"] if isinstance(obs["text_obs"], str) else f"text_obs is embedded: {obs['text_obs'].shape}" 
    if goal_suggestions is not None:
        goal_suggestions = '\n'.join(goal_suggestions)
        title += f"\nGoal suggestions: {goal_suggestions}"
    fig.suptitle(title , fontsize=16,weight="bold" )
    for i, frame in enumerate(obs["obs"][:frame_stack]):
        axs[i].imshow(frame)
        axs[i].axis('off')
        axs[i].set_title(f't={i - frame_stack + 1}', y=-0.15)
    plt.show()

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


