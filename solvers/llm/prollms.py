from random import shuffle
import numpy as np
import os
import re
from tqdm.notebook import tqdm
from .utils import *


def parse_answer(answer, predicate: lambda x: True):
    def auto_type(x):
        return int(x) if abs(int(x) - float(x)) < 1e-7 else round(float(x), 4)
    points = []
    try:
        for numbers in re.findall(r'\((.*?)\)', answer):
            vars = tuple(map(auto_type, numbers.split(',')))
            if predicate(vars):
                points.append(vars)
    except:
        pass
    return points

def create_prompt(observed_points, d=10, n_samples=5):
    prefix = f"""
The following are examples of {d}-dimensional binary control vectors for achieving the corresponding target values. 
Recommend {n_samples} new different {d}-dimensional binary control vectors that can achieve smaller target value.
Your response must only contain the predicted vectors.\n
"""
    suffix = """Examples:\n{Q}\nPredicted Vectors:"""
    prompt_template = PromptTemplate(input_variables=["Q"], template=prefix+suffix)
    prompt = prompt_template.format(Q=set2text(observed_points))
    # print(prompt)
    return prompt    

def sample(model, observed_points, d=10, k_samples=5):
    """
    return: [x1, x2, ..., x_n_samples], where each xi = (xi_1, xi_2, ..., xi_d) - control vector
    """
    prompt = create_prompt(observed_points, d, k_samples)
    answer = model.generate(prompt)
    points = parse_answer(answer, lambda x: True)
    points = [point for point in set(points) if point not in observed_points.keys()]
    return points

class PROLLMS(LLMSolver):
    def __init__(self, problem, budget, k_init=10, k=100, n_samples=5):
        super().__init__(problem, budget, k_init, k, n_samples)

    def sample(self):
        return sample(self.model, self.memory.storage, self.problem.d, self.k_samples)