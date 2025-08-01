import numpy as np
import pandas as pd
from typing import Tuple
from src.core.logic.strategy import Strategy
from src.core.models.asset import Asset
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort


class MovingAverageCrossoverStrategy(Strategy):

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

    def evaluate(self):
        """
        Evaluate the strategy using the latest historical data from the market data port.
        Executes trades via the broker port if a crossover is detected.
        """
        # Fetch historical data for the asset
        data = self.market_data.request_historical_data(self.asset.symbol, None, None)  # Adjust dates as needed
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
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

        # Example: execute trade on crossover
        if current_trend == 1:
            self.price_estimate = current_price * (1 + avg_upturn / 100)
            self.price_std = self.price_estimate * (std_upturn / 100)
            # Example: place a buy order
            self.broker.buy(self.asset, quantity=1)
        else:
            self.price_estimate = current_price * (1 - abs(avg_downturn) / 100)
            self.price_std = self.price_estimate * (std_downturn / 100)
            # Example: place a sell order
            self.broker.sell(self.asset, quantity=1)