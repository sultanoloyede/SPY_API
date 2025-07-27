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
        data = data.groupby("group", group_keys=True).apply(self.assign_turning_point).reset_index(drop=True)

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
            self.price_estimate = current_price * (1 + avg_upturn / 100)
            self.price_std = self.price_estimate * (std_upturn / 100)
        else:
            self.price_estimate = current_price * (1 - abs(avg_downturn) / 100)
            self.price_std = self.price_estimate * (std_downturn / 100)