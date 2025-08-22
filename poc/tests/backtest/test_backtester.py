import pytest
import pandas as pd
from unittest.mock import MagicMock
from poc.backtest.backtester import Backtester
from poc.strategies.random_baseline import RandomBaseline
from poc.backtest.trade_result import TradeResult

def generate_mock_data(hours=168):
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

    # Mock log_trade to avoid printing and capture calls
    backtester.log_trade = MagicMock()

    result = backtester.run()

    # Check top-level result structure
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

    # Check trade objects
    for trade in result["trades"]:
        assert isinstance(trade, TradeResult)
        assert trade.result in ["win", "loss"]
        assert trade.direction in ["buy", "sell"]
        assert isinstance(trade.pips, (int, float))

    assert backtester.log_trade.call_count == result["count"]

    for call in backtester.log_trade.call_args_list:
        kwargs = call.kwargs
        assert "trade_datetime" in kwargs
        assert "result" in kwargs
        assert "units" in kwargs
        assert "direction" in kwargs
        assert "entry_price" in kwargs  # ✅ new check

        assert isinstance(kwargs["trade_datetime"], pd.Timestamp)
        assert kwargs["result"] in [0, 1]
        assert isinstance(kwargs["units"], float)
        assert kwargs["direction"] in ["buy", "sell"]
        assert isinstance(kwargs["entry_price"], (float, int))  # ✅ new check
