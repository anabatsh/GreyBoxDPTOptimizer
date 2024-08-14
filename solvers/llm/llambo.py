from random import shuffle
import numpy as np
import re
from scipy.stats import norm
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from .utils import *


def parse_answer_vector(answer):
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

def parse_answer_value(answer):
    try:
        value = re.findall(r"(-?[\d.]+)", answer)
        if len(value) == 1:
            value = round(float(value[0]), 4)
        else:
            value = None
    except:
        value = None
    return value

def create_prompt_vector(observed_points, candidate_point):
    few_shot_examples = [{'A': y, 'Q': x} for x, y in observed_points.items()]
    example_template = """Control vector: {Q}\nTarget value: {A}"""
    example_prompt = PromptTemplate(input_variables=["Q", "A"], template=example_template)

    prefix = f"""
The following are examples of binary control vectors and the corresponding target values.
For a given binary control vector predict the corresponding target value.
Your response must only contain the predicted target value.
"""
    # , in the format ## target value ##
    suffix = """Control vector: {Q}\nTarget value: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["Q"],
        example_separator=""
    )
    query_example = {'Q': candidate_point}
    few_shot_prompt = few_shot_prompt_template.format(Q=query_example['Q'])
    return few_shot_prompt

def create_prompt_value(observed_points, y_desired):
    few_shot_examples = [{'A': y, 'Q': x} for x, y in observed_points.items()]
    example_template = """Target value: {A}\nControl vector: {Q}"""  
    example_prompt = PromptTemplate(input_variables=["Q", "A"], template=example_template)

    prefix = f"""
The following are examples of binary control vectors for achieving the corresponding target values.
Recommend a binary control vector that can achieve the target value of {y_desired:.6f}.
Your response must only contain the predicted vector.
"""
    # , in the format ## vector ##.
    suffix = """Target value: {A}\nControl vector: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=few_shot_examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["A"],
        example_separator=""
    )
    query_example = {'A': y_desired}
    few_shot_prompt = few_shot_prompt_template.format(A=query_example['A'])
    return few_shot_prompt

def sample(model, observed_points, n_samples=5, alpha=-0.2, n_max_trials=10):
    """
    return: [x1, x2, ..., x_n_samples], where each xi = (xi_1, xi_2, ..., xi_d) - control vector
    """
    # assert alpha >= -1 and alpha <= 1, 'alpha must be between -1 and 1'
    y_best = min(observed_points.values())
    y_worst = max(observed_points.values())
    y_desired = y_best - alpha * np.abs(y_worst - y_best)
    # так промпт каждый раз одинаковый, но можно переместить в цикл и шафлить примеры через сид
    few_shot_prompt = create_prompt_value(observed_points, y_desired)

    trial = 0
    sampled_points = []
    while len(sampled_points) < n_samples and trial < n_max_trials:
        llm_response = model.generate(few_shot_prompt)
        proposed_point = parse_answer_vector(llm_response)
        if proposed_point is not None:
            if proposed_point not in observed_points.keys():
                if proposed_point not in sampled_points:
                    sampled_points.append(proposed_point)
        trial += 1
    return sampled_points

def select(model, observed_points, candidate_points, n_samples=5, n_max_trials=10):
    """
    arguments: []
    return: [x1, x2, ..., x_n_samples], where each xi = (xi_1, xi_2, ..., xi_d) - control vector
    ei - expected improvement (https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html#expected-improvement-ei)
    """
    ei = []
    y_best = min(observed_points.values())

    i_ = np.argmin(observed_points.values())
    x_best = list(observed_points.keys())[i_]
    # print(f'{x_best} | {y_best:.4f} | 0')
    for candidate_point in candidate_points:
        few_shot_prompt = create_prompt_vector(observed_points, candidate_point)

        trial = 0
        predicted_values = []
        while len(predicted_values) < n_samples and trial < n_max_trials:
            llm_response = model.generate(few_shot_prompt)
            predicted_value = parse_answer_value(llm_response)
            if predicted_value is not None:
                predicted_values.append(predicted_value)
            trial += 1

        y_mean = np.mean(predicted_values)
        y_std = np.std(predicted_values)
        if y_std <= 0:
            ei.append(0)
        else:
            delta = -1 * (y_mean - y_best)
            Z = delta / y_std
            ei.append(delta * norm.cdf(Z) + y_std * norm.pdf(Z))

        # print(f'{candidate_point} | {y_mean:.4f} +- {y_std:.4f} | {ei[-1]:.4f}')

    best_point = candidate_points[np.argmax(ei)]
    return [best_point]

class LLAMBO(LLMSolver):
    def __init__(self, problem, budget, k_init=10, k=100, k_samples=5):
        super().__init__(problem, budget, k_init, k, k_samples)

    def sample(self):
        points = sample(self.model, self.memory.storage, self.k_samples)
        if len(points):
            points = select(self.model, self.memory.storage, points, self.k_samples)
        return points