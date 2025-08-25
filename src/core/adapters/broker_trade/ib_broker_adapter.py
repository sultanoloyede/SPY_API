from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.models.asset import Asset
from typing import Optional
from src.core.services.ibapi_client import IBApi
from ibapi.order import Order
from ibapi.contract import Contract
import threading
import time

class IbBrokerAdapter(BrokerTradePort):
    def __init__(self, ib_client: IBApi):
        self.ib_client = ib_client

    def _create_contract(self, asset: Asset) -> Contract:
        contract = Contract()
        contract.symbol = asset.symbol
        contract.secType = "CASH"
        contract.exchange = "IDEALPRO"
        contract.currency = asset.currency
        return contract

    def _create_order(self, action: str, quantity: int) -> Order:
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = "MKT"
        return order

    def buy(self, asset: Asset, quantity: int, price: float) -> str: # Price Parameters is irrelevant when live trading
        contract = self._create_contract(asset)
        order = self._create_order("BUY", quantity)
        order_id = self.ib_client.nextOrderId
        self.ib_client.placeOrder(order_id, contract, order)
        self.ib_client.nextOrderId += 1
        return str(order_id)

    def sell(self, asset: Asset, quantity: int, price: float) -> str: # Price Parameters is irrelevant when live trading
        contract = self._create_contract(asset)
        order = self._create_order("SELL", quantity)
        order_id = self.ib_client.nextOrderId
        self.ib_client.placeOrder(order_id, contract, order)
        self.ib_client.nextOrderId += 1
        return str(order_id)

    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float, action: str) -> str:
        contract = self._create_contract(asset)
        parent_id = self.ib_client.nextOrderId

        parent = self._create_order(action.upper(), quantity, entry_price)
        parent.orderId = parent_id
        parent.transmit = False

        take_profit_order = Order()
        take_profit_order.action = "SELL" if action.upper() == "BUY" else "BUY"
        take_profit_order.orderType = "LMT"
        take_profit_order.totalQuantity = quantity
        take_profit_order.lmtPrice = take_profit
        take_profit_order.parentId = parent_id
        take_profit_order.transmit = False
        take_profit_order.orderId = parent_id + 1

        stop_loss_order = Order()
        stop_loss_order.action = take_profit_order.action
        stop_loss_order.orderType = "STP"
        stop_loss_order.auxPrice = stop_loss
        stop_loss_order.totalQuantity = quantity
        stop_loss_order.parentId = parent_id
        stop_loss_order.transmit = True
        stop_loss_order.orderId = parent_id + 2

        self.ib_client.placeOrder(parent.orderId, contract, parent)
        self.ib_client.placeOrder(take_profit_order.orderId, contract, take_profit_order)
        self.ib_client.placeOrder(stop_loss_order.orderId, contract, stop_loss_order)

        self.ib_client.nextOrderId += 3
        return str(parent_id)
    
    @property
    def value(self):
        return float(self.ib_client.account_summary["NetLiquidation"])

    def compute_stats(self):
        pass # No point in implementing this since IB TWS already gives us the metrics
