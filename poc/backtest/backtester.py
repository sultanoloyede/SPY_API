import pandas as pd
from poc.backtest.trade_result import TradeResult
from poc.config import RR_PRESET, UNIT_PIP_VALUE

class Backtester:
    def __init__(self, strategy, data: pd.DataFrame):
        self.strategy = strategy
        self.data = data.set_index("datetime")
        self.risk_reward = RR_PRESET  # (risk, reward) tuple
        self.pip_unit = UNIT_PIP_VALUE  # e.g. 30 pips = 1 unit

    def run(self, start_date=None, end_date=None):
        df = self.data.copy()

        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        signals = self.strategy.generate_signals(df.reset_index())
        results = []

        for signal in signals:
            entry_time = signal["entry_time"]
            direction = signal["direction"]
            entry_price = signal["entry_price"]

            if entry_time not in df.index:
                continue

            exit_price, exit_time, result, pips = self._simulate_trade(
                df, entry_time, entry_price, direction
            )

            # 🔔 Log each trade to terminal
            self.log_trade(
                trade_datetime=exit_time,
                result=(1 if result == "win" else 0),
                units=pips / self.pip_unit
            )

            results.append(TradeResult(
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                result=result,
                rr_ratio=self.risk_reward[1] / self.risk_reward[0],
                pips=pips
            ))

        return {
            "trades": results,
            "win_pct": self._calculate_win_rate(results),
            "net_units": round(sum(r.pips for r in results) / self.pip_unit, 2),
            "sharpe_ratio": self._calculate_sharpe(results),
            "count": len(results)
        }

    def _simulate_trade(self, df, entry_time, entry_price, direction):
        risk = self.risk_reward[0]
        reward = self.risk_reward[1]
        pip_value = self.pip_unit / 10000  # Convert pip unit to price diff

        # Calculate target and stop loss prices
        if direction == "buy":
            target = entry_price + reward * pip_value
            stop = entry_price - risk * pip_value
        else:
            target = entry_price - reward * pip_value
            stop = entry_price + risk * pip_value

        exit_time = entry_time

        for t, row in df.loc[entry_time:].iterrows():
            price_high = row["high"]
            price_low = row["low"]

            if direction == "buy":
                if price_high >= target:
                    return target, t, "win", reward * self.pip_unit
                if price_low <= stop:
                    return stop, t, "loss", -risk * self.pip_unit
            else:
                if price_low <= target:
                    return target, t, "win", reward * self.pip_unit
                if price_high >= stop:
                    return stop, t, "loss", -risk * self.pip_unit

            exit_time = t

        # No hit: assume loss at last close
        return row["close"], exit_time, "loss", -risk * self.pip_unit

    def _calculate_win_rate(self, trades):
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.result == "win")
        return round(100 * wins / len(trades), 2)

    def _calculate_sharpe(self, trades):
        if not trades:
            return 0.0

        returns = [r.pips / self.pip_unit for r in trades]
        mean_return = pd.Series(returns).mean()
        std_return = pd.Series(returns).std()

        if std_return == 0 or pd.isna(std_return):
            return 0.0

        return round(mean_return / std_return, 2)

    def log_trade(self, trade_datetime, result, units):
        result_str = "WIN" if result == 1 else "LOSS"
        print(f"[{trade_datetime}] → {result_str} | Units: {'+' if result == 1 else '-'}{abs(units)}")
