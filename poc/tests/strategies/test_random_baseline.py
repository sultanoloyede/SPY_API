import pytest
import pandas as pd
from poc.strategies.random_baseline import RandomBaseline



def test_random_baseline_signal_output_structure():
    """
    Tests that RandomBaseline generates a valid list of trade signals.
    """
    # Create 1 week of hourly data
    data = pd.DataFrame({
        "datetime": pd.date_range(start="2023-01-01", periods=168, freq="h"),  # 7*24 = 168
        "open": [1.0] * 168,
        "high": [1.1] * 168,
        "low": [0.9] * 168,
        "close": [1.0] * 168,
    })

    strategy = RandomBaseline()
    signals = strategy.generate_signals(data)

    assert isinstance(signals, list), "Output should be a list"

    for signal in signals:
        assert isinstance(signal, dict), "Each signal should be a dictionary"
        assert "entry_time" in signal
        assert "entry_price" in signal
        assert "direction" in signal
        assert signal["direction"] in ["buy", "sell"]
        assert isinstance(signal["entry_price"], float)
