from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.models.asset import Asset
from typing import Optional
from ib_insync import IB, MarketOrder, LimitOrder, Stock, Order, BracketOrder

class IbBrokerAdapter(BrokerTradePort):
    def __init__(self, ib_client):
        self.ib_client = ib_client

    def _create_contract(self, asset: Asset):
        return Stock(asset.symbol, exchange=asset.exchange, currency=asset.currency)

    def buy(self, asset: Asset, quantity: int, price: Optional[float] = None) -> str:
        # TODO: Implement IB API buy logic
        contract = self._create_contract(asset)
        order = (
            LimitOrder('BUY', quantity, price)
            if price is not None else
            MarketOrder('BUY', quantity)
        )
        trade = self.ib_client.placeOrder(contract, order)
        return trade.order.permId  # IB's permanent order ID as str

    def sell(self, asset: Asset, quantity: int, price: Optional[float] = None) -> str:
        # TODO: Implement IB API sell logic
        contract = self._create_contract(asset)
        order = (
            LimitOrder('SELL', quantity, price)
            if price is not None else
            MarketOrder('SELL', quantity)
        )
        trade = self.ib_client.placeOrder(contract, order)
        return trade.order.permId

    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float, action: str) -> str:
        # TODO: Implement IB API bracket order logic

        contract = self._create_contract(asset)

        bracket = BracketOrder(
            action=action.upper(),  # Make sure it's "BUY" or "SELL"
            totalQuantity=quantity,
            limitPrice=entry_price,
            takeProfitPrice=take_profit,
            stopLossPrice=stop_loss
        )

        for order in bracket:
            self.ib_client.placeOrder(contract, order)

        return bracket[0].permId  # Return parent order ID
