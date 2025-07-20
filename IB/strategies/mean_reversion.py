import numpy as np
from typing import Tuple
from strategies.strategy import Strategy

class MeanReversionStrategy(Strategy):
    def evaluate(self, data) -> Tuple[float, float]:
        prices = data["close"].values
        estimation = np.mean(prices)
        std_dev = np.std(prices)
        return estimation, std_dev
