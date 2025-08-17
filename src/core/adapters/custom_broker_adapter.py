from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.models.asset import Asset
from src.utils.logger import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CustomBrokerAdapter(BrokerTradePort):
    def __init__(self, initial_cash: float):
        self._initial_balance = initial_cash
        self.current_balance = self._initial_balance
        self.trades = []
        self.closed_trades = []
        logger.info(f"Initialized CustomBrokerAdapter with starting cash: {self._initial_balance}")

    def buy(self, asset: Asset, quantity: int, price: float) -> None:
        if price is None:
            logger.error("Buy failed: No price provided for asset.")
            raise ValueError("Asset must have a 'price' attribute for backtesting.")

        total_cost = price * quantity
        if total_cost > self.current_balance:
            logger.warning(f"Buy failed: Insufficient funds to buy {quantity} of {asset.symbol} at ${price:.2f}")
            raise ValueError()
        
        trade = {
            'type': 'BUY',
            'asset': asset,
            'quantity': quantity,
            'buy_price': price,
            'status': 'OPEN'
        }
        self.trades.append(trade)
        self.current_balance -= total_cost
        logger.info(f"Executed BUY: {quantity}x {asset.symbol} at ${price:.2f} | Remaining Cash: ${self.current_balance:.2f}")
        return f"BUY-{len(self.trades)-1}"

    def sell(self, asset: Asset, quantity: int, price: float) -> str:
        for trade in self.trades:
            if trade['asset'].symbol == asset.symbol and trade['status'] == 'OPEN' and trade['quantity'] >= quantity:
                trade['sell_price'] = price
                trade['sell_quantity'] = quantity
                trade['status'] = 'CLOSED'
                self.closed_trades.append(trade)
                self.current_balance += price * quantity
                logger.info(f"Executed SELL: {quantity}x {asset.symbol} at ${price:.2f} | New Cash: ${self.current_balance:.2f}")

        logger.warning(f"Sell failed: No open trade found for {asset.symbol}")

    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float, action: str) -> str:
        action = action.upper()
        if action == "BUY":
            if entry_price * quantity > self.current_balance:
                logger.warning(f"Bracket order failed: Insufficient funds to buy {quantity} of {asset.symbol}")
                raise ValueError()
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
            self.current_balance -= entry_price * quantity
            logger.info(f"Placed BUY_BRACKET for {asset.symbol}: Entry ${entry_price}, TP ${take_profit}, SL ${stop_loss}")

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
            self.current_balance += entry_price * quantity
            logger.info(f"Placed SELL_BRACKET for {asset.symbol}: Entry ${entry_price}, TP ${take_profit}, SL ${stop_loss}")

        logger.error(f"Bracket order failed: Invalid action '{action}'")

    def _get_open_trades(self):
        open_trades = [tr for tr in self.trades if tr['status'] == 'OPEN']
        logger.debug(f"Open trades count: {len(open_trades)}")
        return open_trades

    def _get_closed_trades(self):
        logger.debug(f"Closed trades count: {len(self.closed_trades)}")
        return self.closed_trades
    
    def compute_stats(self):
        """
        Log summary statistics about trades using the logger.
        """
        closed_trades = [tr for tr in self.trades if tr['status'] == 'CLOSED']
        num_trades = len(closed_trades)
        num_wins = len([tr for tr in closed_trades if tr.get('sell_price', 0) > tr.get('buy_price', 0)])
        num_losses = num_trades - num_wins
        total_profit = sum([(tr.get('sell_price', 0) - tr.get('buy_price', 0)) * tr.get('quantity', 1) for tr in closed_trades])
        win_rate = (num_wins / num_trades) * 100.0 if num_trades > 0 else 0.0
        perc_return = (total_profit * 100.0 / self.current_balance) if self.current_balance > 0 else 0.0
        # Buy and hold calculation (simple version)
        if closed_trades:
            initial_buy = closed_trades[0].get('buy_price', 0)
            final_sell = closed_trades[-1].get('sell_price', 0)
            perc_buy_hold_return = ((final_sell - initial_buy) * 100.0 / initial_buy) if initial_buy > 0 else 0.0
        else:
            perc_buy_hold_return = 0.0

        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Winning Trades: {num_wins}")
        logger.info(f"Losing Trades: {num_losses}")
        logger.info(f"Return [%]: {perc_return:.2f}%")
        logger.info(f"Buy and Hold Return [%]: {perc_buy_hold_return:.2f}%")
        logger.info(f"Total Profit: ${total_profit:.2f}")
        logger.info(f"Initial Capital: ${self.current_balance:.2f}")
