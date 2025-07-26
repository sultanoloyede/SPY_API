import backtrader as bt
from datetime import datetime

from trading_pipeline import TradingStrategy  # Ensure this matches your actual strategy class name

class TestStrategy(TradingStrategy):
    pass  # Inherit everything from your pipeline strategy

def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)

    # Example: Load data from CSV (adjust path and format as needed)
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2020, 1, 1),
        todate=datetime(2021, 1, 1)
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(10000)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

if __name__ == "__main__":
    run_backtest()