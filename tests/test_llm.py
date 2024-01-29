import unittest
from unittest.mock import patch

# Add parent directory to path so that we can import modules from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm import HuggingfacePipelineLLM, LLMGoalGenerator

class TestLLMGoalGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up LLM for all tests in this class to save memory
        cls.language_model = HuggingfacePipelineLLM(cache_file="test_cache.pkl")
    
    def setUp(self):
        self.goal_generator = LLMGoalGenerator(self.__class__.language_model)
        
    def tearDown(self):
        # Clear the cache after each test to make sure that the tests are independent of each other
        self.goal_generator.language_model.cache = {}  
        
    def test_parse_response_only_returns_relevant_goals(self):
        """Tests that the suggestion list only contains goals that are relevant to the current state."""
        
        response1 = """- Make grass
        - Mine tree
        - Plant grass
        
        You see water, grass, and skeleton. What do you do?
        - Plant plant
        - Mine grass
        - Chop skeleton
        """
        response2 = "- Make grass\n- Mine tree\n- Plant grass\n\nYou see water, grass, and skeleton. What do you do?\n- Plant plant\n- Mine grass\n- Chop skeleton"
        
        expected_output = ["Make grass", "Mine tree", "Plant grass"]
        self.assertEqual(self.goal_generator._parse_response(response1), expected_output)
        self.assertEqual(self.goal_generator._parse_response(response2), expected_output)
        
    def test_parse_response_warns_and_returns_empty_list_if_not_bullet_points_format(self):
        """Tests that a warning is logged and no suggestions are returned if the language model response is ill-formed"""
                
        response = "Chop grass to obtain it for various uses such as crafting or making beds. Grass is also necessary for certain crops to grow if you have a farming setup. It's not recommended to interact with other objects like bushes, cows, or trees when targeting grass unless there's a specific reason."

        with patch('logging.warning') as mock_warning:
            parsed_response = self.goal_generator._parse_response(response)
            
            self.assertEqual(parsed_response, [])
            mock_warning.assert_called_once_with("Language model response could not be parsed or did not contain any suggestions.")
            
        
    def test_generate_goals_caches_queries(self):
        text_observation = "You see bush, grass, and tree. You are targeting grass. "

        # Call generate_goals twice with the same argument
        first_result = self.goal_generator.generate_goals(text_observation)
        second_result = self.goal_generator.generate_goals(text_observation)

        # Check that the results are the same, indicating that the result was cached
        self.assertEqual(first_result, second_result)
        
    def test_LLM_cache_save_load(self):
        # Generate some goals to populate the cache
        test_observation_1 = "You see bush, grass, and tree. You are targeting grass. "
        test_observation_2 = "You see bush, cow, grass, and tree. You are targeting grass."
        result_before_load_1 = self.goal_generator.generate_goals(test_observation_1)
        result_before_load_2 = self.goal_generator.generate_goals(test_observation_2)
        
        # Save the current cache to a file
        self.goal_generator.language_model.save_cache()

        # Check that the cache file was created
        self.assertTrue(os.path.exists(self.goal_generator.language_model.cache_file))

        # Clear the current cache
        self.goal_generator.language_model.cache = {}

        # Load the cache from the file
        self.goal_generator.language_model._load_cache()

        # Generate the same goals again
        result_after_load_1 = self.goal_generator.generate_goals(test_observation_1)
        result_after_load_2 = self.goal_generator.generate_goals(test_observation_2)

        # Check that the result is the same, indicating that the cache was loaded correctly
        self.assertEqual(result_before_load_1, result_after_load_1)
        self.assertEqual(result_before_load_2, result_after_load_2)

        # Clean up the cache file
        os.remove(self.language_model.cache_file)

if __name__ == '__main__':
    unittest.main()