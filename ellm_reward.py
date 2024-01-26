import random

def compute_ellm_reward(action_name, goal_suggestions):
    # Compute cosine similarity between action name and each of the goal suggestions, return max and closest suggestion
    # TODO: Implement this, currently returns random similarity between 0 and 1 and random suggestion
    return random.random(), random.choice(goal_suggestions)
