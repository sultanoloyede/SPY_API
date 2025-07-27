import numpy as np
import pandas as pd
from typing import Tuple
from strategies.strategy import Strategy

class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, ib, contract):
        super().__init__()
        self.ib = ib
        self.contract = contract
        self.order_id = 100

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["moving_avg_50"] = data["close"].rolling(window=50).mean()
        data["moving_avg_200"] = data["close"].rolling(window=200).mean()
        data["trend"] = (data["moving_avg_50"] > data["moving_avg_200"]).astype(int)
        data["group"] = (data["trend"] != data["trend"].shift()).cumsum()
        return data

    def assign_turning_point(self, group: pd.DataFrame) -> pd.DataFrame:
        first_close = group["close"].iloc[0]
        turning_value = group["close"].max() if group["trend"].iloc[0] == 1 else group["close"].min()
        pct_diff = ((turning_value - first_close) / first_close) * 100

        group["turning_point"] = turning_value
        group["turning_diff_pct"] = pct_diff
        return group

    def evaluate(self, data: pd.DataFrame) -> Tuple[float, float]:
        data = self.calculate_indicators(data)
        data = data.groupby("group").apply(self.assign_turning_point).reset_index(drop=True)

        is_first_in_group = (data["trend"] != data["trend"].shift())
        first_rows = data[is_first_in_group]

        upturns = first_rows[first_rows["trend"] == 1]["turning_diff_pct"]
        downturns = first_rows[first_rows["trend"] == 0]["turning_diff_pct"]

        avg_upturn = upturns.mean()
        std_upturn = upturns.std()
        avg_downturn = downturns.mean()
        std_downturn = downturns.std()

        current_price = data["close"].iloc[-1]
        current_trend = data["trend"].iloc[-1]

        if current_trend == 1:
            price_target = current_price * (1 + avg_upturn / 100)
            price_deviation = price_target * (std_upturn / 100)
        else:
            price_target = current_price * (1 - abs(avg_downturn) / 100)
            price_deviation = price_target * (std_downturn / 100)

        return price_target, price_deviation

    def generate_signal(self, data: pd.DataFrame) -> str:
        data = self.calculate_indicators(data)
        if len(data) < 200:
            return "HOLD"

        ma50 = data["moving_avg_50"].iloc[-1]
        ma200 = data["moving_avg_200"].iloc[-1]
        prev_ma50 = data["moving_avg_50"].iloc[-2]
        prev_ma200 = data["moving_avg_200"].iloc[-2]

        if prev_ma50 < prev_ma200 and ma50 > ma200:
            return "BUY"
        elif prev_ma50 > prev_ma200 and ma50 < ma200:
            return "SELL"
        return "HOLD"

    def on_new_data(self, data: pd.DataFrame):
        signal = self.generate_signal(data)
        price_target, price_deviation = self.evaluate(data)

        if signal == "BUY":
            print(f"Signal: BUY — Target: {price_target:.4f} ± {price_deviation:.4f}")
            orders = self.ib.bracketOrder(
                parentOrderId=self.order_id,
                action="BUY",
                quantity=100,
                profitTarget=price_target,
                stopLoss=price_target - price_deviation
            )
            for o in orders:
                self.ib.placeOrder(o.orderId, self.contract, o)
            self.order_id += 3

        elif signal == "SELL":
            print(f"Signal: SELL — Target: {price_target:.4f} ± {price_deviation:.4f}")
            orders = self.ib.bracketOrder(
                parentOrderId=self.order_id,
                action="SELL",
                quantity=100,
                profitTarget=price_target,
                stopLoss=price_target + price_deviation
            )
            for o in orders:
                self.ib.placeOrder(o.orderId, self.contract, o)
            self.order_id += 3
