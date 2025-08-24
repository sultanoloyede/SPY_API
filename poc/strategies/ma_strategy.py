import pandas as pd
from poc.strategies.base_strategy import BaseStrategy

class MAStrategy(BaseStrategy):
    def __init__(self, short_window=50, long_window=200):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> list[dict]:
        """
        Generate buy/sell signals using MA crossovers.
        Buy  = golden cross (short crosses above long)
        Sell = death cross  (short crosses below long)
        """
        df = data.copy()
        df["short_ma"] = df["close"].rolling(window=self.short_window).mean()
        df["long_ma"] = df["close"].rolling(window=self.long_window).mean()

        df = df.dropna(subset=["short_ma", "long_ma"])

        signals = []

        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            # Golden cross (buy)
            if prev["short_ma"] <= prev["long_ma"] and curr["short_ma"] > curr["long_ma"]:
                signals.append({
                    "entry_time": curr["datetime"],
                    "entry_price": curr["close"],
                    "direction": "buy"
                })

            # Death cross (sell)
            elif prev["short_ma"] >= prev["long_ma"] and curr["short_ma"] < curr["long_ma"]:
                signals.append({
                    "entry_time": curr["datetime"],
                    "entry_price": curr["close"],
                    "direction": "sell"
                })

        return signals
