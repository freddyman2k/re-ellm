import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common import logger

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
    'env_reward': None,  # to be specified later
    'embedding_shape': (384,),
    'seed': 1,
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

def make_env(name='CrafterTextEnv-v1',
                action_space_type='harder',
                env_reward=None,
                device='cpu',  
                use_language_state=False,
                use_sbert=False,
                frame_stack=4,
                **kwargs):
    env = gym.make(name,
                action_space_type=action_space_type,
                env_reward=env_reward,
                device=device,  
                use_language_state=use_language_state,
                use_sbert=use_sbert,)
    env = CustomFrameStack(env, frame_stack)
    return env

def embed_text_observation(obs, obs_embedder):
    """Embed text observations manually because this makes using stable baselines policy and replay buffer easier"""
    return {
        'obs': obs['obs'], 
        'text_obs': obs_embedder.embed(obs['text_obs'])
        }

def evaluate(agent, env, obs_embedder, n_episodes=10):
    total_reward = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:   
            embedded_state = embed_text_observation(state, obs_embedder)
            action, _ = agent.predict(embedded_state)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
    return total_reward / n_episodes

def train_agent(max_env_steps=5000000, eval_every=5000, log_every=1000):
    language_model = HuggingfacePipelineLLM("mistralai/Mistral-7B-Instruct-v0.2", cache_file="cache.pkl")
    # language_model = DummyLLM() # for debugging other parts that do not need GPU, if you use this you don't need to submit a job to the cluster
    goal_generator = LLMGoalGenerator(language_model=language_model)
    env = make_env(**env_spec)
    env = TransformObsSpace(env) # Transform observation space to be compatible with stable baselines later
    obs_embedder = TextEmbedder()
    reward_calculator = ELLMRewardCalculator()
    agent = DQN('MultiInputPolicy', env, verbose=1)
    # Configure the logger (do not remove, necessary for stable baselines to work)
    new_logger = logger.configure('./logs', ['stdout', 'log', 'csv', 'tensorboard'])
    agent.set_logger(new_logger)
    
    prev_achieved_goals = []
    
    global_step = 0
    state, _ = env.reset()
    embedded_state = embed_text_observation(state, obs_embedder)
    done = False
    last_eval_step = 0
    last_log_step = 0
    elapsed_episodes = 0

    while global_step < max_env_steps:        
        # Generate goal suggestions, filtering achieved ones
        goal_suggestions = goal_generator.generate_goals(state['text_obs'])
        #TODO: Maybe also filter out suggestions that are too similar to each other, e.g. "eat plant" and "eat plants"   
        goal_suggestions = [goal for goal in goal_suggestions if goal not in prev_achieved_goals]

        # Interact with the environment
        action, _ = agent.predict(embedded_state)  
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Embed text observations manually because this makes using stable baselines policy and replay buffer easier
        embedded_next_state = embed_text_observation(next_state, obs_embedder)

        # Compute suggestion achievement reward
        action_name = env.get_action_name(action)
        intrinsic_reward, closest_suggestion = reward_calculator.compute_cosine_similarity(action_name, goal_suggestions)
        if intrinsic_reward > SIMILARITY_THRESHOLD and info["action_success"]:
            reward = reward + intrinsic_reward
            # If the action was successful and the LLM made a suggestion that corresponds to it, add it to the list of achieved goals
            prev_achieved_goals.append(closest_suggestion)
        
        # Update agent using any RL algorithm 
        agent.replay_buffer.add(embedded_state, embedded_next_state, action, reward, float(done), [info])
        agent.train(1, batch_size=BATCH_SIZE)
        
        if done:
            elapsed_episodes += 1
            if global_step - last_log_step >= log_every:
                agent.logger.record("ours/elapsed_episodes", elapsed_episodes)
                agent.logger.dump(global_step)
                last_log_step = global_step
                
            if global_step - last_eval_step >= eval_every:
                # Evaluate agent
                eval_reward = evaluate(agent, env, obs_embedder)
                agent.logger.record("ours/mean_eval_reward", eval_reward)
                # Remember time of last evaluation
                last_eval_step = global_step
                
            
            # Store language model cache on disk for future runs. TODO: Probably do this less frequently, for debug purposes done here
            language_model.save_cache()
            
            # Reset environment
            state = env.reset()
            done = False

        state = next_state
        embedded_state = embedded_next_state
        global_step += 1

if __name__ == "__main__":    
    train_agent()
    