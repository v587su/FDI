import random 
random.seed(42)

class PromptInject:
    def __init__(self, prompt):
        self.poison_prompt = prompt
        print(len(self.poison_prompt))

    def inject_one(self, item):
        # item: (query, solution)
        return (item[0], item[1]+self.poison_prompt)
    
    def inject_into_code(self, code):
        return code+self.poison_prompt

        