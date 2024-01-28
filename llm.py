class LLMGoalGenerator:
    def __init__(self):  
        self.prev_achieved_goals = []
        
    def generate_goals(self, text_observation):
        # TODO: Generate suggestions, dummy suggestion for now
        goal_suggestions = ["overthink"]

        return goal_suggestions