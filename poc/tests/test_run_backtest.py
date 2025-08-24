import pytest
import pandas as pd
from unittest.mock import patch
from poc.run_backtest import main

def generate_mock_data(n=1000):
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": [1.0] * n,
        "high": [1.1] * n,
        "low": [0.9] * n,
        "close": [1.0 + 0.0001 * (i % 10) for i in range(n)],
    })

@patch("poc.run_backtest.get_forex_data")
def test_main_runs_without_error(mock_data_fetch):
    mock_data_fetch.return_value = generate_mock_data()
    
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")
