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
        estimation = np.mean(prices)
        std_dev = np.std(prices)
        return estimation, std_dev

    def generate_signal(self, data: pd.DataFrame) -> str:
        estimation, std_dev = self.evaluate(data)
        current_price = data["close"].iloc[-1]

        if current_price < estimation - std_dev:
            return "BUY"
        elif current_price > estimation + std_dev:
            return "SELL"
        return "HOLD"

    def on_new_data(self, data: pd.DataFrame):
        signal = self.generate_signal(data)

        if signal == "BUY":
            print("Signal: BUY - submitting bracket order")
            # Initializing order object
            orders = self.ib.bracketOrder(
                parentOrderId=self.order_id,
                action="BUY",
                quantity=100,
                profitTarget=1.17760,
                stopLoss=1.17700
            )
            for o in orders:
                self.ib.placeOrder(o.orderId, self.contract, o)
            self.order_id += 3  

        elif signal == "SELL":
            print("Signal: SELL - submitting bracket order")
            # Initializing order object
            orders = self.ib.bracketOrder(
                parentOrderId=self.order_id,
                action="SELL",
                quantity=100,
                profitTarget=1.17700,
                stopLoss=1.17760
            )
            for o in orders:
                self.ib.placeOrder(o.orderId, self.contract, o)
            self.order_id += 3 # Order ID increment
