import pytest
import pandas as pd
from unittest.mock import patch
from poc.backtest.compare_strategies import compare_to_baseline
from poc.strategies.ma_strategy import MAStrategy

def generate_mock_data(n=500):
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": [1.0] * n,
        "high": [1.1] * n,
        "low": [0.9] * n,
        "close": [1.0 + 0.0001 * (i % 10) for i in range(n)],  # slight variation to create crosses
    })

@patch("poc.strategies.random_baseline.RandomBaseline.generate_signals")
def test_compare_to_baseline_returns_expected_keys(mock_baseline_signals):
    # Mock RandomBaseline output to return a fixed trade
    mock_baseline_signals.return_value = [
        {"entry_time": pd.Timestamp("2024-01-01 12:00:00"), "entry_price": 1.0, "direction": "buy"}
    ]

    data = generate_mock_data()
    strategy = MAStrategy(short_window=3, long_window=5)

    results = compare_to_baseline(data, strategy, name="TestMAStrat")

    assert "custom" in results
    assert "baseline" in results

    for key in ["win_pct", "net_units", "sharpe_ratio", "count", "trades"]:
        assert key in results["custom"]
        assert key in results["baseline"]
