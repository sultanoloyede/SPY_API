import numpy as np
from typing import Tuple
from strategies.strategy import Strategy
import pandas as pd

class MeanReversionStrategy(Strategy):
    def __init__(self, ib, contract):
        super().__init__()
        self.ib = ib 
        self.contract = contract  # Contract to trade
        self.order_id = 100  # Identification val for order

    def evaluate(self, data) -> Tuple[float, float]:
        prices = data["close"].values
        self.price_estimate = np.mean(prices)
        self.price_std = np.std(prices)