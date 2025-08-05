import pandas as pd
from poc.strategies.ma_strategy import MAStrategy

def generate_mock_crossing_data():
    """
    Generates a DataFrame where a golden and death cross occurs.
    Short MA will cross above and below Long MA.
    """
    data = []
    for i in range(250):
        # First 125: flat → no cross
        if i < 125:
            close_price = 1.0
        # Next 65: slight uptrend → golden cross
        elif i < 190:
            close_price = 1.0 + 0.001 * (i - 125)
        # Final 60: downtrend → death cross
        else:
            close_price = 1.14 - 0.001 * (i - 190)

        data.append(close_price)

    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=250, freq="h"),
        "open": data,
        "high": [p + 0.001 for p in data],
        "low": [p - 0.001 for p in data],
        "close": data,
    })

def test_ma_strategy_generates_expected_signals():
    df = generate_mock_crossing_data()
    strategy = MAStrategy(short_window=5, long_window=20)
    signals = strategy.generate_signals(df)

    assert isinstance(signals, list)
    assert len(signals) > 0

    for signal in signals:
        assert "entry_time" in signal
        assert "entry_price" in signal
        assert signal["direction"] in ["buy", "sell"]
