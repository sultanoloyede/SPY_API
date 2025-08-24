from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[dict]:
        """
        Each signal is a dict like:
        {
            'entry_time': pd.Timestamp,
            'entry_price': float,
            'direction': 'buy' | 'sell'
        }
        """
        pass
