from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.models.asset import Asset
import backtrader as bt

class BacktraderApp():
    def __init__(self):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(1000000) # Initial ammount of cash
    
class BacktraderBrokerAdapter(BrokerTradePort):
    def __init__(self, bt_client: BacktraderApp):
        self.bt_client = bt_client

    def buy(self, asset: Asset, quantity: int, price: float = None) -> str:
        # Find or create the Backtrader data feed for the asset
        data = self._get_data_feed(asset)
        # Place a market or limit buy order
        if price is None:
            order = self.bt_client.cerebro.broker.buy(data=data, size=quantity)
        else:
            order = self.bt_client.cerebro.broker.buy(data=data, size=quantity, price=price, exectype=bt.Order.Limit)
        return str(order.ref)  # Return order reference

    def sell(self, asset: Asset, quantity: int, price: float = None) -> str:
        data = self._get_data_feed(asset)
        if price is None:
            order = self.bt_client.cerebro.broker.sell(data=data, size=quantity)
        else:
            order = self.bt_client.cerebro.broker.sell(data=data, size=quantity, price=price, exectype=bt.Order.Limit)
        return str(order.ref)

    def bracket_order(self, asset: Asset, quantity: int, entry_price: float, take_profit: float, stop_loss: float) -> str:
        data = self._get_data_feed(asset)
        bracket = self.bt_client.cerebro.broker.bracket(
            data=data,
            size=quantity,
            price=entry_price,
            stopprice=stop_loss,
            limitprice=take_profit
        )
        return str(bracket[0].ref)  # Return main order reference

    def _get_data_feed(self, asset: Asset):
        # You need to implement this to return the correct Backtrader data feed for the asset
        # For example, search self.bt_client.cerebro.datas for the matching symbol
        for data in self.bt_client.cerebro.datas:
            if hasattr(data, 'symbol') and data.symbol == asset.symbol:
                return data
        raise ValueError(f"Data feed for asset {asset.symbol} not found")
