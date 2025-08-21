from src.utils.logger import logger
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.core.models.bar import Bar
from src.core.logic.strategy import Strategy
from src.utils.config import RISK_FREE_RATE

import threading
from datetime import datetime

import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort, strategies: list[Strategy]):
        self.broker = broker
        self.market_data = market_data
        self.strategies: list[Strategy] = strategies
        self.bar_data:list[Bar] = []
        self.portfolio_value: dict[datetime, int] = {}

    def run(self, asset: Asset, threaded: bool = True):
        def engine_loop():
            while True:

                # Blocks until new bar is available in multithreaded applications
                bar = self.market_data.next_bar(asset)
                
                # When market data is completed, compute statistics and end this function
                if bar is None:
                    self.broker.compute_stats()
                    break

                # Add bar to list data and update portfolio value
                self.bar_data.append(bar)
                self.portfolio_value[bar.timestamp] = self.broker.value

                # Iterate over strategies to generate signals
                for strategy in self.strategies:
                    strategy.evaluate(self.bar_data)

        if threaded:
            t = threading.Thread(target=engine_loop, daemon=True)
            t.start()
            return t
        else:
            engine_loop()

    def generate_data_plot(self):
        dates = [bar.timestamp for bar in self.bar_data]
        opens = [bar.open for bar in self.bar_data]
        highs = [bar.high for bar in self.bar_data]
        lows = [bar.low for bar in self.bar_data]
        closes = [bar.close for bar in self.bar_data]

        equity_dates = list(self.portfolio_value.keys())
        equity_values = list(self.portfolio_value.values())

        # Compute daily returns from equity curve manually
        returns = []
        for i in range(1, len(equity_values)):
            prev = equity_values[i - 1]
            curr = equity_values[i]
            if prev != 0:
                returns.append((curr - prev) / prev)
            else:
                returns.append(0.0)

        # Mean return
        mean_return = sum(returns) / len(returns) if returns else 0.0

        # Standard deviation
        std_return = 0.0
        if returns:
            mean = mean_return
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5

        sharpe_ratio = 0.0
        if std_return > 0:
            sharpe_ratio = (mean_return - RISK_FREE_RATE) / std_return

        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[f"{self.market_data.asset} Price", "Account Equity"]
        )

        # Top: Candlestick chart
        fig.add_trace(go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="Asset Price"
        ), row=1, col=1)

        # Bottom: Equity curve
        fig.add_trace(go.Scatter(
            x=equity_dates,
            y=equity_values,
            mode="lines",
            name="Account Equity",
            line=dict(color="cyan", width=2)
        ), row=2, col=1)

        fig.update_layout(
            title=f"Historical Data & Account Equity - {self.market_data.asset} | Sharpe Ratio: {sharpe_ratio:.4f}",
            xaxis_title="Date",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        fig.update_yaxes(title_text=f"{self.market_data.asset} Price ({self.market_data.asset.currency})", row=1, col=1)
        fig.update_yaxes(title_text=f"Account Equity ({self.market_data.asset.currency})", row=2, col=1)
        py.offline.plot(fig)