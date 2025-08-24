import pandas as pd
from poc.strategies.rsi_strategy import RSIStrategy

def generate_mock_rsi_data():
    prices = [1.00]*10 + [1.01, 1.02, 1.03, 1.04, 1.05, 1.06] + [1.05, 1.04, 1.03, 1.02, 1.01, 1.00]
    return pd.DataFrame({
        "datetime": pd.date_range(start="2024-01-01", periods=len(prices), freq="h"),
        "open": prices,
        "high": [p + 0.01 for p in prices],
        "low": [p - 0.01 for p in prices],
        "close": prices
    })

def test_rsi_strategy_generates_signals():
    df = generate_mock_rsi_data()
    strategy = RSIStrategy(rsi_period=5, oversold_threshold=30, overbought_threshold=70)

    signals = strategy.generate_signals(df)

    assert isinstance(signals, list)
    for signal in signals:
        assert "entry_time" in signal
        assert "entry_price" in signal
        assert signal["direction"] in ["buy", "sell"]
