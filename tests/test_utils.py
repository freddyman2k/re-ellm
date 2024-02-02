import unittest
from datetime import datetime


# Add parent directory to path so that we can import modules from the parent directory
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import make_exp_name

class TestMakeExpName(unittest.TestCase):
    def test_make_exp_name(self):
        env_spec = {
            'name': 'CrafterTextEnv-v1',
            'action_space_type': 'harder',
            'env_reward': False, 
            'seed': 1,
            'dying': True,
            'length': 400,
            'similarity_threshold': .99,
            'check_ac_success': False,
            'novelty_bias': False,
            'goal_generator': "LLMGoalGenerator",
            'language_model': "testllm",
            'frame_stack': 4,
        }

        exp_name = make_exp_name(env_spec)
        date_time_str = datetime.now().strftime("%Y-%m-%d_%H")

        expected_exp_name = f"run_seed-1_LLMGoalGenerator_testllm_no-novelty-bias_ignore-ac-success_{date_time_str}"
        self.assertEqual(exp_name, expected_exp_name)

if __name__ == '__main__':
    unittest.main()