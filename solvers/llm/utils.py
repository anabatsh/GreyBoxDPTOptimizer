from random import shuffle
import re
import os
import json
import numpy as np
from tqdm.notebook import tqdm
from openai import OpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from ..utils import * 

API_KEY = "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"



class Memory:
    def __init__(self, randomness=0, size=100):
        self.storage = {}
        self.size = size
        self.randomness = randomness

    def update(self, points):
        self.storage |= points

        if len(self.storage) > self.size:
            random_cap = int(self.randomness * self.size)
            best_cap = self.size - random_cap

            best = [
                vars for (vars, value) in sorted(
                    self.storage.items(), key=lambda x: x[1]
                )[:best_cap]
            ]

            if random_cap:
                random = [k for k in self.storage if k not in best]
                shuffle(random)
                best.extend(random[:random_cap])

            self.storage = {
                k: self.storage[k] for k in best
            }

class ChatGPT():
    def __init__(self, version='gpt-3.5-turbo'):
        self.version = version
        self.client = OpenAI(api_key=API_KEY)
        self.system_content = """You are trying to minimize unknown black-box function."""

    # All you can do is to provide input values and observe the output. 
    # You must not repeat already known points.

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.version,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

class LLMSolver():
    def __init__(self, problem, budget, k_init=10, k_samples=5, k_memory=100):
        super().__init__()
        self.model = ChatGPT()
        self.memory = Memory(size=k_memory)

    def update(self, points, targets):
        points, targets = super().update(points, targets)
        self.memory.update({tuple(point): round(target, 4) for point, target in zip(points, targets)})