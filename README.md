# Onyx Trading Engine

A modular and extensible trading strategy creation platform built with scalable design patterns, supporting live trading (Interactive Brokers) and backtesting (custom/Yahoo adapters) for stocks, forex, and more.

## Features
- **Live Trading:** Connects to Interactive Brokers (IB or IBkr) via IB API and Traders Works Station (TWS) for real-time execution.
- **Backtesting:** Simulate strategies using historical data from Yahoo Finance or custom CSVs.
- **Strategy Framework:** Easily implement and swap trading strategies (e.g., Moving Average Crossover).
- **Thread-Safe Data Flow:** Robust handling of live and historical bar data using queues and event-driven callbacks.
- **Comprehensive Logging:** Unified logging for all trades, statistics, and system events.
- **Visualization:** Candlestick charts with buy/sell markers for backtest results.

## Quick Start

To setup the environment, read and follow [`src/README.md`](src/README.md)


## Architecture Overview

![System Level View](Resources/Images/System/SystemLevelView.png)

- **Domain Models:** `Asset`, `Bar` (stocks, forex, crypto, etc.)
- **Ports:** `MarketDataPort`, `BrokerTradePort` (abstract interfaces)
- **Adapters:** IB API, Yahoo Finance, Custom Backtest
- **Strategies:** Modular, plug-and-play (e.g., Moving Average, Kalman Filter)
- **Engine:** Orchestrates data, strategy, and broker actions

## Example: Moving Average Crossover Strategy

```python
from src.core.logic.trading_engine import TradingEngine
from src.core.adapters.yf_market_adapter import YFMarketDataAdapter
from src.core.logic.moving_average import MovingAverageCrossoverStrategy
from src.core.adapters.custom_broker_adapter import CustomBrokerAdapter
from src.core.models.bar import Bar
from src.core.models.asset import Asset, AssetType
from datetime import datetime, timedelta

if __name__ == "__main__":
    asset = Asset(AssetType.FOREX, "EUR", "USD")
    market_data_adapter = YFMarketDataAdapter(asset)
    market_data_adapter.request_historical_data(asset, datetime.today()-timedelta(365*4), datetime.today())
    broker_adapter = CustomBrokerAdapter(10000)
    strategy = MovingAverageCrossoverStrategy(broker_adapter, asset)
    trading_engine = TradingEngine(broker_adapter, market_data_adapter, [strategy])
    trading_engine.run(asset, threaded=False)

```

## Visualization

![Backtest Chart](Resources/Images/Charts/IBPaperTrading.png)

Candlestick charts with buy/sell markers are generated for backtest results.

## Authors
- [Josue Dazogbo](https://github.com/JDazogbo)
- [Sultan Oloyede](https://github.com/sultanoloyede)
- [Jedidiah Ange-Emmanuel Kouakou](https://github.com/ARelaxedScholar)
- [Racine Kane](https://github.com/Racine-04)

## Resources
- [System Architecture](Resources/Design/SystemArchitecture.md)
- [Logging & Backtest Charts](Resources/Images/Charts/IBPaperTrading.png)
- [System Level View](Resources/Images/System/SystemLevelView.png)