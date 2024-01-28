from environment import CustomFrameStack, TransformObsSpace
from llm import LLMGoalGenerator
from policy import DQNPolicy
from ellm_reward import ELLMRewardCalculator
import text_crafter.text_crafter
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common import logger



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


def evaluate(policy, env, n_episodes=10):
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            # TODO: Embed text observations
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def train_agent(max_env_steps=5000000, eval_every=5000):
    goal_generator = LLMGoalGenerator()
    env = make_env(**env_spec)
    reward_calculator = ELLMRewardCalculator()
    policy = DQNPolicy(env.observation_space.shape, env.action_space.n)
    
    global_step = 0
    state = env.reset()
    done = False
    last_eval_episode = 0

    while global_step < max_env_steps:        
        # Generate k suggestions, filtering achieved ones
        goal_suggestions = goal_generator.generate_goals(state['text'])

        # Interact with the environment
        obs = {
            'image_state': state['image'], #TODO: rename'image', 'text'
            'text_state': state['text'],
            'goal': " ".join(goal_suggestions)
        }
        # TODO: Embed text observations
        action = policy.select_action(obs)  
        next_state, reward, done, info = env.step(action)

        # Compute suggestion achievement reward
        action_name = env.get_action_name(action)
        intrinsic_reward, closest_suggestion = reward_calculator.compute_cosine_similarity(action_name, goal_suggestions)
        if intrinsic_reward > SIMILARITY_THRESHOLD and info["action_success"]:
            reward = reward + intrinsic_reward
            # If the action was successful and the LLM made a suggestion that corresponds to it, add it to the list of achieved goals
            goal_generator.prev_achieved_goals.append(closest_suggestion)
        
        # Update agent using any RL algorithm 
        policy.buffer.store_transition(state, action, reward, next_state, done)
        policy.update(BATCH_SIZE)
        
        if done:
            if global_step - last_eval_episode >= eval_every:
                # Evaluate agent
                eval_reward = evaluate(policy, env)
                # Remember time of last evaluation
                last_eval_episode = global_step
            
            # Reset environment
            state = env.reset()
            done = False

        state = next_state  
        global_step += 1

if __name__ == "__main__":
    env = make_env(**env_spec)
    env = TransformObsSpace(env)
    obs, info = env.reset()
    print(obs)
    obs, reward, terminated, truncated, info = env.step(0)
    model = DQN('MultiInputPolicy', env, verbose=1)
    # Configure the logger
    new_logger = logger.configure('./logs', ['stdout', 'log', 'csv', 'tensorboard'])
    model.set_logger(new_logger)
    
    
    train_agent()