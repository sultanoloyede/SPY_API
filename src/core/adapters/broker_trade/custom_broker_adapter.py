from src.core.models.bar import Bar
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.utils.logger import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CustomBrokerAdapter(BrokerTradePort):
    def __init__(self, initial_cash: float, market_data_adapter: MarketDataPort):
        self._initial_balance = initial_cash
        self._current_balance = self._initial_balance

        self.market_port: MarketDataPort = market_data_adapter

        self.trades = []
        self.closed_trades = []
        logger.info(f"Initialized CustomBrokerAdapter with starting cash: {self._initial_balance}")

    def buy(self, asset: Asset, quantity: int, price: float) -> None:

        total_cost = price * quantity
        if total_cost <= self._current_balance:
            trade = {
                'type': 'BUY',
                'asset': asset,
                'quantity': quantity,
                'buy_price': price,
                'status': 'OPEN'
            }
            self.trades.append(trade)
            self._current_balance -= total_cost
            logger.info(f"Executed BUY: {quantity}x {asset.symbol} at ${price:.2f} | Remaining Cash: ${self._current_balance:.2f}")
        else:
            logger.warning(f"Buy failed: Insufficient funds to buy {quantity} of {asset.symbol} at ${price:.2f}")

    def sell(self, asset: Asset, quantity: int, price: float) -> str:
        for trade in self.trades:
            if trade['asset'].symbol == asset.symbol and trade['status'] == 'OPEN' and trade['quantity'] >= quantity:
                trade['sell_price'] = price
                trade['sell_quantity'] = quantity
                trade['status'] = 'CLOSED'
                self.closed_trades.append(trade)
                self._current_balance += price * quantity
                logger.info(f"Executed SELL: {quantity}x {asset.symbol} at ${price:.2f} | New Cash: ${self._current_balance:.2f}")

        logger.warning(f"Sell failed: No open trade found for {asset.symbol}")

    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float, action: str) -> str:
        action = action.upper()
        if action == "BUY":
            if entry_price * quantity < self._current_balance:
                trade = {
                    'type': 'BUY_BRACKET',
                    'asset': asset,
                    'quantity': quantity,
                    'buy_price': entry_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'status': 'OPEN'
                }
                self.trades.append(trade)
                self._current_balance -= entry_price * quantity
                logger.info(f"Placed BUY_BRACKET for {asset.symbol}: Entry ${entry_price}, TP ${take_profit}, SL ${stop_loss}")
            else:
                logger.warning(f"Bracket order failed: Insufficient funds to buy {quantity} of {asset.symbol}")

        elif action == "SELL":
            trade = {
                'type': 'SELL_BRACKET',
                'asset': asset,
                'quantity': quantity,
                'sell_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'status': 'OPEN'
            }
            self.trades.append(trade)
            self._current_balance += entry_price * quantity
            logger.info(f"Placed SELL_BRACKET for {asset.symbol}: Entry ${entry_price}, TP ${take_profit}, SL ${stop_loss}")

        logger.error(f"Bracket order failed: Invalid action '{action}'")

    def _get_open_trades(self):
        open_trades = [trade for trade in self.trades if trade['status'] == 'OPEN']
        logger.debug(f"Open trades count: {len(open_trades)}")
        return open_trades

    def _get_closed_trades(self):
        logger.debug(f"Closed trades count: {len(self.closed_trades)}")
        return self.closed_trades
    
    @property
    def value(self):
        # Start with current cash balance
        total_value = self._current_balance
        # Add value of open sell trades (liquid asset value)
        for trade in self.trades:
            if trade['type'] == 'BUY' and trade['status'] == 'OPEN':
                # Use the most recent price for the asset
                price = self.market_port.current_bar.close
                quantity = trade.get('sell_quantity', trade.get('quantity', 0))
                if price is not None:
                    total_value += price * quantity
        return total_value

    def compute_stats(self):

        # Normal statistics
        closed_trades = [trade for trade in self.trades if trade['status'] == 'CLOSED']
        num_trades = len(closed_trades)
        num_wins = len([trade for trade in closed_trades if trade.get('sell_price', 0) > trade.get('buy_price', 0)])
        num_losses = num_trades - num_wins
        total_profit = self.value - self._initial_balance
        win_rate = (num_wins / num_trades) * 100.0 if num_trades > 0 else 0.0
        perc_return = (total_profit * 100.0 / self._initial_balance) if self.value > 0 else 0.0

        # Buy and Hold return rate
        if closed_trades:
            initial_buy = closed_trades[0].get('buy_price', 0)
            final_sell = closed_trades[-1].get('sell_price', 0)
            perc_buy_hold_return = ((final_sell - initial_buy) * 100.0 / initial_buy) if initial_buy > 0 else 0.0
        else:
            perc_buy_hold_return = 0.0

        # Printing Results
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Winning Trades: {num_wins}")
        logger.info(f"Losing Trades: {num_losses}")
        logger.info(f"Return [%]: {perc_return:.2f}%")
        logger.info(f"Buy and Hold Return [%]: {perc_buy_hold_return:.2f}%")
        logger.info(f"Total Profit: ${total_profit:.2f}")
        logger.info(f"Initial Capital: ${self._initial_balance:.2f}")
