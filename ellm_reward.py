import random
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util
import torch

class ELLMRewardCalculator:
    def __init__(self, model_name='paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_cosine_similarity(self, action_description: str, goal_suggestions: List[str]) -> Tuple[float, str]:
        """Computes the cosine similarity between the action description and each goal suggestion. 
        Returns the highest scoring goal suggestion and its score.
        
        Args:
            action_description (str): The action description
            goal_suggestions (list[str]): A list of goal suggestions to compare the action description to.
            
        Returns:
            float: The cosine similarity score of the highest scoring goal suggestion
            str: The highest scoring goal suggestion
        """
        
        # If there are no goal suggestions, return a similarity score of 0 and no closest suggestion
        if len(goal_suggestions) == 0:
            return 0, None
        
        # Embed the action description and goal suggestions
        embeddings = self.model.encode([action_description] + goal_suggestions, convert_to_tensor=True)

        # Compute cosine similarity between action and each goal
        cosine_scores = util.cos_sim(embeddings[0], embeddings[1:])

        # Extract the highest scoring goal suggestion and its score
        max_score, max_idx = torch.max(cosine_scores, dim=1)
        closest_suggestion = goal_suggestions[max_idx.item()]

        return max_score.item(), closest_suggestion

    
