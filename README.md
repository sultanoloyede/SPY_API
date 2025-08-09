# Black Wealth Systems

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

- Latest [IBApi version](https://interactivebrokers.github.io/downloads/TWS%20API%20Install%201030.01.msi)

## Setting up the environment

```
python -m venv .venv # This may vary depending on python environment variable setup. Try python, python3, python3.13.
.\.venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```
