from typing import List, Tuple
import random


class Model:
    def __init__(self, categories: List[str]) -> None:
        self.categories = categories
    
    def classify(self, text: str) -> Tuple[str, float]:
        return random.choice(self.categories), random.uniform(0, 1)
