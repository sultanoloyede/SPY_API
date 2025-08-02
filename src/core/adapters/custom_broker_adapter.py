from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.models.asset import Asset

class CustomBrokerAdapter(BrokerTradePort):

    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.trades = []
        self.closed_trades = []

    def buy(self, asset: Asset, quantity: int, price:float) -> str:
        if price is None:
            raise ValueError("Asset must have a 'price' attribute for backtesting.")
        total_cost = price * quantity
        if total_cost > self.cash:
            return "Insufficient funds"
        trade = {
            'type': 'BUY',
            'asset': asset,
            'quantity': quantity,
            'buy_price': price,
            'status': 'OPEN'
        }
        self.trades.append(trade)
        self.cash -= total_cost
        return f"BUY-{len(self.trades)-1}"

    def sell(self, asset: Asset, quantity: int, price:float) -> str:
        for trade in self.trades:
            if trade['asset'].symbol == asset.symbol and trade['status'] == 'OPEN' and trade['quantity'] >= quantity:
                trade['sell_price'] = price
                trade['sell_quantity'] = quantity
                trade['status'] = 'CLOSED'
                self.closed_trades.append(trade)
                self.cash += price * quantity
                return f"SELL-{self.trades.index(trade)}"
        return "No open trade found to sell"

    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float, action: str) -> str:
        # Simulate a bracket order: open trade, close at take_profit or stop_loss
        if action.upper() == "BUY":
            if entry_price * quantity > self.cash:
                return "Insufficient funds"
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
            self.cash -= entry_price * quantity
            return f"BUY_BRACKET-{len(self.trades)-1}"
        elif action.upper() == "SELL":
            # For simplicity, assume we have the asset to sell
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
            self.cash += entry_price * quantity
            return f"SELL_BRACKET-{len(self.trades)-1}"
        else:
            return "Invalid action for bracket order"

    def get_open_trades(self):
        return [tr for tr in self.trades if tr['status'] == 'OPEN']

    def get_closed_trades(self):
        return