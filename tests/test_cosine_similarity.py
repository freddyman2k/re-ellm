import unittest

# Add parent directory to path so that we can import modules from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ellm_reward import ELLMRewardCalculator

class TestELLMRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.reward_calculator = ELLMRewardCalculator()

    def test_compute_cosine_similarity_returns_expected_closest_goal_and_score(self):
        # Example from the paper 
        action_name = "Chop tree"
        goal_suggestions = ["Cut down the tree", "Dig in the grass", "Attack the cow"]
        expected_closest_goal = "Cut down the tree" 
        similarity_score, closest_goal = self.reward_calculator.compute_cosine_similarity(action_name, goal_suggestions)

        self.assertEqual(closest_goal, expected_closest_goal)
        self.assertTrue(-1 <= similarity_score <= 1)  # Similarity score should be between -1 and 1
        
    def test_compute_cosine_similarity_with_empty_goal_suggestions(self):
        action_name = "Chop tree"
        goal_suggestions = []
        expected_closest_goal = None
        expected_similarity_score = 0
        similarity_score, closest_goal = self.reward_calculator.compute_cosine_similarity(action_name, goal_suggestions)

        self.assertEqual(closest_goal, expected_closest_goal)
        self.assertEqual(similarity_score, expected_similarity_score)


if __name__ == '__main__':
    unittest.main()