import pandas as pd
from poc.backtest.backtester import Backtester
from poc.strategies.random_baseline import RandomBaseline

def compare_to_baseline(data: pd.DataFrame, strategy, name="Custom", start_date=None, end_date=None):
    print(f"Backtesting: {name}")
    strat_result = Backtester(strategy, data).run(start_date, end_date)

    print(f"Backtesting: RandomBaseline")
    baseline_result = Backtester(RandomBaseline(), data).run(start_date, end_date)

    print("\nSTRATEGY COMPARISON")
    print(f"{'Metric':<15} | {name:<15} | RandomBaseline")
    print("-" * 48)
    for metric in ["count", "win_pct", "net_units", "sharpe_ratio"]:
        print(f"{metric:<15} | {strat_result[metric]:<15} | {baseline_result[metric]}")

    return {
        "custom": strat_result,
        "baseline": baseline_result
    }
