import pytest
import pandas as pd
from poc.backtest.backtester import Backtester
from poc.strategies.random_baseline import RandomBaseline
from poc.backtest.trade_result import TradeResult

def generate_mock_data(hours=168):
    """
    Generate mock OHLCV data for N hours (default: 1 week)
    """
    return pd.DataFrame({
        "datetime": pd.date_range(start="2024-01-01", periods=hours, freq="h"),
        "open": [1.0] * hours,
        "high": [1.1] * hours,
        "low": [0.9] * hours,
        "close": [1.0] * hours,
    })

def test_backtester_runs_and_returns_correct_structure():
    data = generate_mock_data()
    strategy = RandomBaseline()
    backtester = Backtester(strategy, data)

    result = backtester.run()

    # Check top-level keys
    assert isinstance(result, dict)
    assert "trades" in result
    assert "win_pct" in result
    assert "net_units" in result
    assert "sharpe_ratio" in result
    assert "count" in result

    # Check types
    assert isinstance(result["trades"], list)
    assert isinstance(result["win_pct"], float)
    assert isinstance(result["net_units"], float)
    assert isinstance(result["sharpe_ratio"], float)
    assert isinstance(result["count"], int)

    # Validate individual trade objects
    for trade in result["trades"]:
        assert isinstance(trade, TradeResult)
        assert trade.result in ["win", "loss"]
        assert trade.direction in ["buy", "sell"]
        assert isinstance(trade.pips, (int, float))

