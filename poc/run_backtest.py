from poc.data.fetch_twelve_data import get_forex_data
from poc.strategies.ma_strategy import MAStrategy
from poc.strategies.rsi_strategy import RSIStrategy
from poc.backtest.compare_strategies import compare_to_baseline
from poc.config import START_DATE, END_DATE

def main():
    pair = "EUR/USD"
    data = get_forex_data(pair, output_size=1500)

    strategy = MAStrategy(short_window=50, long_window=200)

    compare_to_baseline(data, strategy, name="MAStrategy", start_date=START_DATE, end_date=END_DATE)

    # Initialize your strategy
    # strategy = RSIStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70)

    # compare_to_baseline(data, strategy, name="RSIStrategy", start_date=START_DATE, end_date=END_DATE)

if __name__ == "__main__":
    main()
