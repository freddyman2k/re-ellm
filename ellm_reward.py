import random

from sentence_transformers import SentenceTransformer, util
import torch

class ELLMRewardCalculator:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_cosine_similarity(self, action_description, goal_suggestions):
        # Embed the action description and goal suggestions
        embeddings = self.model.encode([action_description] + goal_suggestions, convert_to_tensor=True)

        # Compute cosine similarity between action and each goal
        cosine_scores = util.cos_sim(embeddings[0], embeddings[1:])

        # Extract the highest scoring goal suggestion and its score
        max_score, max_idx = torch.max(cosine_scores, dim=1)
        closest_suggestion = goal_suggestions[max_idx.item()]

        return max_score.item(), closest_suggestion

    
