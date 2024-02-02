import gymnasium as gym
import numpy as np

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
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)
        info['text_obs'] = obs['text_obs']
        return obs, info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.observation(obs)
        info['text_obs'] = obs['text_obs']
        return obs, reward, terminated, truncated, info
    
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

