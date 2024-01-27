from environment import TextCrafterEnv
from llm import LLMGoalGenerator
from policy import DQNPolicy
from ellm_reward import ELLMRewardCalculator

SIMILARITY_THRESHOLD = 0.99
BATCH_SIZE = 64

def evaluate(policy, env, n_episodes=10):
    total_reward = 0
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
    return total_reward / n_episodes

def train_agent(max_env_steps=5000000, eval_every=5000):
    env = TextCrafterEnv()
    goal_generator = LLMGoalGenerator()
    reward_calculator = ELLMRewardCalculator()
    policy = DQNPolicy(env.observation_space.shape, env.action_space.n)
    
    global_step = 0
    prev_achieved_goals = []
    
    state = env.reset()
    done = False
    last_eval_episode = 0

    while global_step < max_env_steps:        
        # Generate k suggestions, filtering achieved ones
        goal_suggestions = goal_generator.generate_goals(state['text'])
        # TODO: Maybe filter out suggestions that are too similar to each other, e.g. "eat plant" and "eat plants"
        goal_suggestions = [goal for goal in goal_suggestions if goal not in prev_achieved_goals]

        # Interact with the environment
        obs = {
            'image_state': state['image'],
            'text_state': state['text'],
            'goal': " ".join(goal_suggestions)
        }
        action = policy.select_action(obs)  
        next_state, reward, done, info = env.step(action)

        # Compute suggestion achievement reward
        action_name = env.get_action_name(action)
        intrinsic_reward, closest_suggestion = reward_calculator.compute_cosine_similarity(action_name, goal_suggestions)
        if intrinsic_reward > SIMILARITY_THRESHOLD and info["action_success"]:
            reward = reward + intrinsic_reward
            # If the action was successful and the LLM made a suggestion that corresponds to it, add it to the list of achieved goals
            prev_achieved_goals.append(closest_suggestion)
        
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
    train_agent()