class LLMGoalGenerator:
    def __init__(self, only_new_goals=True):  
        self.prev_achieved_goals = []
        self.only_new_goals = only_new_goals
        
    def generate_goals(self, text_observation):
        # TODO: Generate suggestions, dummy suggestion for now
        goal_suggestions = ["overthink"]

        #TODO: Maybe filter out suggestions that are too similar to each other, e.g. "eat plant" and "eat plants"
        if self.only_new_goals:      
            goal_suggestions = [goal for goal in goal_suggestions if goal not in self.prev_achieved_goals]
        return goal_suggestions