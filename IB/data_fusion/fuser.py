from abc import ABC, abstractmethod
from typing import List
from strategies.strategy import Strategy

class Fuser(ABC):
    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies

    @abstractmethod
    def predict(self):
        """
        Calls evaluate on all strategies and fuses their outputs.
        """
        pass