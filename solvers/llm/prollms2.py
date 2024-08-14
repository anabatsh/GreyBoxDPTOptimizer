from random import shuffle
import numpy as np
import re


def parse_answer(answer):
    def auto_type(x):
        return int(x) if abs(int(x) - float(x)) < 1e-7 else round(float(x), 4)
    vector = None
    try:
        vector = re.findall(r"\((.*)\)", answer)
        if len(vector) == 1:
            vector = tuple(map(auto_type, vector[0].split(',')))
    except:
        pass
    return vector

def create_prompt(observed_points):
    few_shot_examples = [{'A': f'{y: .4f}', 'Q': x} for x, y in observed_points.items()]
    example_template = """Target value: {A}\nControl vector: {Q}\n"""  
    example_prompt = PromptTemplate(input_variables=["Q", "A"], template=example_template)

    prefix = """
The following are examples of binary control vectors for achieving the corresponding target values.
Recommend a binary control vector that can achieve even smaller target value.
Your response must only contain the predicted vector.
"""
    suffix = """Target value: {A}\nControl vector: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["A"],
        example_separator="\n"
    )
    query_example = {'A': 'possible minimum'}
    few_shot_prompt = few_shot_prompt_template.format(A=query_example['A'])
    # print(few_shot_prompt)
    return few_shot_prompt

def sample(model, observed_points, n_samples=5, n_max_trials=10):
    """
    return: [x1, x2, ..., x_n_samples], where each xi = (xi_1, xi_2, ..., xi_d) - control vector
    """
    few_shot_prompt = create_prompt(observed_points)

    trial = 0
    sampled_points = []
    while len(sampled_points) < n_samples and trial < n_max_trials:
        llm_response = model.generate(few_shot_prompt)
        # print(f'answer {trial + 1}', llm_response)
        proposed_point = parse_answer(llm_response)
        if proposed_point is not None:
            if proposed_point not in observed_points.keys():
                if proposed_point not in sampled_points:
                    sampled_points.append(proposed_point)
        trial += 1
    return sampled_points
    
class PROLLMS2(LLMSolver):
    def __init__(self, problem, budget, k_init=10, k=100, k_samples=5):
        super().__init__(problem, budget, k_init, k, k_samples)

    def sample(self):
        return sample(self.model, self.memory.storage, self.k_samples)
    
    