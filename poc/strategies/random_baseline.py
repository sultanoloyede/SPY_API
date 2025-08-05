import random
import pandas as pd
from poc.strategies.base_strategy import BaseStrategy

class RandomBaseline(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> list[dict]:
        """
        Randomly enters between 0 and 9 trades per week.
        Each trade occurs at a random timestamp and chooses buy/sell at random.

        Args:
            data (pd.DataFrame): OHLCV data with 'datetime' column.

        Returns:
            List[dict]: Trade signals with random direction and timing
        """
        signals = []

        # Make sure datetime is in datetime format
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime")

        # Group by weekly segments
        weekly_groups = data.groupby(pd.Grouper(freq="W"))

        for _, week_df in weekly_groups:
            if len(week_df) == 0:
                continue

            # Random number of trades for this week
            num_trades = random.randint(0, 9)

            if num_trades == 0:
                continue

            # Sample random timestamps without replacement
            sampled_rows = week_df.sample(n=min(num_trades, len(week_df)), replace=False)

            for dt, row in sampled_rows.iterrows():
                signal = {
                    "entry_time": dt,
                    "entry_price": row["close"],
                    "direction": random.choice(["buy", "sell"]),
                }
                signals.append(signal)

        # Sort by time to keep consistent order
        signals.sort(key=lambda x: x["entry_time"])
        return signals
