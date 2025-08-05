import pytest
import pandas as pd
from poc.data.fetch_twelve_data import get_forex_data


def test_get_forex_data_structure():
    """
    Test that get_forex_data returns a DataFrame with the correct structure.
    """

    # Fetch a small amount of data to avoid hitting rate limits
    df = get_forex_data("EUR/USD", output_size=10)

    # 1. Check the result is a DataFrame
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"

    # 2. Check required columns exist
    required_columns = {"datetime", "open", "high", "low", "close"}
    assert required_columns.issubset(set(df.columns)), f"Missing expected columns: {required_columns - set(df.columns)}"

    # 3. Ensure we have at least one row
    assert len(df) > 0, "No data returned from API"

    # 4. Check datetime order
    assert df["datetime"].is_monotonic_increasing, "Datetime column is not sorted"
