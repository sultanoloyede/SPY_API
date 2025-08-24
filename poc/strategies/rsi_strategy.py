import pandas as pd
from poc.strategies.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    def __init__(self, rsi_period=14, oversold_threshold=30, overbought_threshold=70):
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        df = df.copy()
        df["price_change"] = df["close"].diff()
        df["gain"] = df["price_change"].clip(lower=0)
        df["loss"] = -df["price_change"].clip(upper=0)

        avg_gain = df["gain"].rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = df["loss"].rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / (avg_loss + 1e-10)  # avoid divide by zero
        df["rsi"] = 100 - (100 / (1 + rs))

        signals = []

        for i in range(1, len(df)):
            prev_rsi = df.iloc[i - 1]["rsi"]
            curr_rsi = df.iloc[i]["rsi"]
            entry_price = df.iloc[i]["open"]
            entry_time = df.iloc[i]["datetime"]

            # Oversold → Buy
            if prev_rsi < self.oversold_threshold and curr_rsi >= self.oversold_threshold:
                signals.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "direction": "buy"
                })
            # Overbought → Sell
            elif prev_rsi > self.overbought_threshold and curr_rsi <= self.overbought_threshold:
                signals.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "direction": "sell"
                })

        return signals
