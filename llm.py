from abc import ABC, abstractmethod
import logging
import pickle
import os
import random
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

TEXTCRAFTER_PROMPT_PREFIX = """Valid actions: sleep, eat, attack, chop, drink, place, make, mine
You are a player playing a game. Suggest the best actions the player can take based on the things you see and the items in your inventory. Only use valid actions and objects.

You see plant, tree, and skeleton. You are targeting skeleton. What do you do?
- Eat plant
- Chop tree
- Attack skeleton

You see water, grass, cow, and diamond. You are targeting grass. You have in your inventory plant. What do you do?
- Drink water
- Chop grass
- Attack cow
- Place plant

"""
TEXTCRAFTER_PROMPT_SUFFIX = "What do you do?\n"
MAX_NEW_TOKENS = 50

class LLMBaseClass(ABC):
    """A wrapper for a language model that can generate text given a prompt."""
    
    def __init__(self, cache_file="cache.pkl", max_cache_size=1000):
        self.max_cache_size = max_cache_size
        self.cache_file = cache_file
        if os.path.exists(self.cache_file):
            self._load_cache()
        else:
            self.cache = {}
    
    def generate_response(self, prompt: str, use_cache=True) -> str:
        """Generates a response given a prompt, using a cache lookup instead if possible. 

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        if prompt in self.cache and use_cache:
            return self.cache[prompt]

        response = self._generate_response_impl(prompt)
        if len(self.cache) < self.max_cache_size:
            # Only add to cache if it is not full yet, otherwise always we will need to perform inference again
            # TODO: Would be nice to have a cache that automatically removes the oldest or least used entries instead
            self.cache[prompt] = response
        return response        
    
    def _generate_response_impl(self, prompt: str) -> str:
        """Actual implementation of response generation. This method should be implemented by subclasses.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
        raise NotImplementedError
    
    def _load_cache(self):
        """Loads the cache from disk."""
        with open(self.cache_file, "rb") as f:
            self.cache = pickle.load(f)
    
    def save_cache(self):
        """Saves the cache to disk."""
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)

class HuggingfacePipelineLLM(LLMBaseClass):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", use_lower_precision=True, cache_file="cache.pkl"):
        """Initializes a huggingface text generation pipeline.

        Args:
            model_name (str): The name of the pre-trained model to use.
            use_lower_precision (bool): Whether to use lower precision settings for the model. Necessary for some models to fit on the GPU.
        """
        
        super().__init__(cache_file)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_lower_precision:
            # quantization setup adapted from: https://www.datacamp.com/tutorial/mistral-7b-tutorial
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            self.pipe = pipeline(
                "text-generation", 
                model=model, 
                tokenizer=tokenizer, 
                device_map="auto"
            )
        
    def _generate_response_impl(self, prompt: str) -> str:
        return self.pipe(prompt, 
                         do_sample=True, 
                         max_new_tokens=MAX_NEW_TOKENS, 
                         return_full_text=False
                         )[0]['generated_text']
        

class LLMGoalGenerator:
    """Prompts an LLM with a text description of the current state the agent is in, to generate a list of suggested actions for the agent to pursue."""
    def __init__(self, language_model: LLMBaseClass): 
        self.language_model = language_model
        
    def generate_goals(self, text_observation: str, prompt_prefix=TEXTCRAFTER_PROMPT_PREFIX, prompt_suffix=TEXTCRAFTER_PROMPT_SUFFIX) -> List[str]: 
        """Generates a list of suggested actions for the agent to pursue.
        
        Args:
            text_observation (str): A text description of the current state the agent is in.
            
        Returns:
            list[str]: A list of suggested actions for the agent to pursue.
        """

        # Generate a response from the language model
        full_prompt = prompt_prefix + text_observation + prompt_suffix
        response = self.language_model.generate_response(full_prompt)
            
        suggestion_list = self._parse_response(response)
        
        # Log a warning if no suggestions were generated, this could indicate a problem with the language model
        if len(suggestion_list) == 0:
            logging.warning(f"Language model response could not be parsed or did not contain any suggestions. Trying again...")
            
            # Try again without using the cache
            response2 = self.language_model.generate_response(full_prompt, use_cache=False)
            suggestion_list = self._parse_response(response2)
            
            if len(suggestion_list) == 0:
                logging.warning(f"Language model response could not be parsed again or did not contain any suggestions. \
                    Prompt:\n {full_prompt}\nResponse 1:\n {response}\nResponse 2:\n {response2}\n Returning empty list.")
            
        return suggestion_list
    
    def _parse_response(self, response: str) -> List[str]:
        """Parses a language model response to extract the suggested actions.
        
        Args:
            response (str): The language model response to parse.
            
        Returns:
            list[str]: A list of suggested actions for the agent to pursue.
        """
        
        suggestion_list = []
        for line in response.splitlines():
            line_no_whitespace = line.strip()
            if line_no_whitespace.startswith("-"):
                suggestion_list.append(line_no_whitespace[2:])
            else:
                # Stop parsing when we reach the end of the suggestions 
                # (LLM may generate more text after the suggestions for the current text obs)
                break            
        return suggestion_list

class ConstantGoalGenerator:
    """A goal generator that always suggests a predefined list of goals."""
    def __init__(self, goal_list):
        self.goal_list = goal_list
        
    def generate_goals(self, text_observation: str, prompt_prefix=TEXTCRAFTER_PROMPT_PREFIX, prompt_suffix=TEXTCRAFTER_PROMPT_SUFFIX) -> List[str]: 
        """Generates a list of suggested actions for the agent to pursue.
        
        Args:
            text_observation (str): A text description of the current state the agent is in.
            
        Returns:
            list[str]: A list of suggested actions for the agent to pursue.
        """
        return self.goal_list

class ConstantSamplerGoalGenerator:
    """A goal generator that samples from a predefined list of goals."""
    def __init__(self, goal_list):
        self.goal_list = goal_list
        
    def generate_goals(self, text_observation: str, prompt_prefix=TEXTCRAFTER_PROMPT_PREFIX, prompt_suffix=TEXTCRAFTER_PROMPT_SUFFIX) -> List[str]: 
        """Generates a list of suggested actions for the agent to pursue.
        
        Args:
            text_observation (str): A text description of the current state the agent is in.
            
        Returns:
            list[str]: A list of suggested actions for the agent to pursue.
        """
        return [random.choice(self.goal_list)]

### Dummy language models for testing purposes ###
 
class TestCacheLLM(LLMBaseClass):
    """A dummy language model for testing purposes that returns only cached responses."""
    def __init__(self, cache_file="cache.pkl"):
        super().__init__(cache_file=cache_file)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logging.warning("TestCacheLLM is intended for debugging on the CPU. Are you sure you want to use this model?")
        
    def _generate_response_impl(self, prompt: str) -> str:
        return ""
    
    def save_cache(self):
        """Since we do not generate new responses, this model should does not need to write to the cache."""
        logging.warning("TestCacheLLM.save_cache() was called, but this model does not generate \
            new responses and thus cannot write to the cache. Are you sure you want to use this model?")
        pass


class ConstantLLM(LLMBaseClass):
    """A dummy language model for testing purposes that always returns the same response."""
    def __init__(self, response="- Chop tree"):
        super().__init__(cache_file="")
        self.response = response
        
    def _generate_response_impl(self, prompt: str) -> str:
        return self.response
    
    def save_cache(self):
        """Since we do not generate new responses, this model should does not need to write to the cache."""
        logging.warning("ConstantLLM.save_cache() was called, but this model does not generate \
            new responses and does cannot write to the cache. Are you sure you want to use this model?")
        pass