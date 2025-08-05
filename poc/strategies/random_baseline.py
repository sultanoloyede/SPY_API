import random
import pandas as pd
from poc.strategies.base_strategy import BaseStrategy

class RandomBaseline(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> list[dict]:
        """
        Randomly enters between 0 and 9 trades per week.
        Ensures no duplicate timestamps are used.
        """
        signals = []
        used_datetimes = set()

        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime")

        weekly_groups = data.groupby(pd.Grouper(freq="W"))

        for _, week_df in weekly_groups:
            if len(week_df) == 0:
                continue

            available_rows = week_df[~week_df.index.isin(used_datetimes)]
            available_rows = available_rows[~available_rows.index.duplicated(keep="first")]

            num_trades = random.randint(0, min(9, len(available_rows)))

            if num_trades == 0:
                continue

            sampled_rows = available_rows.sample(n=num_trades, replace=False)

            for dt, row in sampled_rows.iterrows():
                used_datetimes.add(dt)
                signals.append({
                    "entry_time": dt,
                    "entry_price": row["close"],
                    "direction": random.choice(["buy", "sell"]),
                })

        signals.sort(key=lambda x: x["entry_time"])
        return signals
