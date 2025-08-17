from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import BarData as IB_BAR
from ibapi.account_summary_tags import AccountSummaryTags
from ibapi.contract import *
from threading import Thread
from src.core.models.bar import Bar
import threading
import time
from queue import Queue
from src.utils.logger import logger

class IBApi(EWrapper, EClient):
    _instance = None  # Class-level singleton holder

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(IBApi, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Prevent __init__ from re-running on subsequent instantiations
        if hasattr(self, '_initialized') and self._initialized:
            return
        EClient.__init__(self, self)
        self.nextOrderId = None
        self.account_summary = {}
        self.order_id_ready = threading.Event()
        self.historical_data_buffer: Queue[Bar] = Queue()
        self.historical_data_done = threading.Event()
        self.connect_and_run()
        self._initialized = True  # Flag to prevent re-initialization

    def nextValidId(self, orderId: int):
        self.nextOrderId = orderId
        self.order_id_ready.set()

    def get_next_order_id(self):
        self.order_id_ready.wait()
        order_id = self.nextOrderId
        self.nextOrderId += 1
        return order_id

    def historicalData(self, reqId: int, bar: IB_BAR) -> None:
        bar = Bar(
            timestamp=bar.date,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume
        )
        self.historical_data_buffer.put(bar)

    def historicalDataUpdate(self, reqId:int, bar:IB_BAR) -> None:
        bar = Bar(
            timestamp=bar.date,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume
        )
        self.historical_data_buffer.put(bar)
        logger.debug(f"Processed new bar: {bar}")

    def accountSummary(self, reqId, account, tag, value, currency):
        self.account_summary[tag] = value

    def accountSummaryEnd(self, reqId: int):
        logger.debug(f"ANSWER accountSummaryEnd. Account Value: {self.account_summary["NetLiquidation"]}")

    def connect_and_run(self, host="127.0.0.1", port=7497, client_id=1):
        self.connect(host, port, client_id)
        thread = Thread(target=self.run, daemon=True)
        thread.start()

        # Optional: Wait until connection is confirmed
        time.sleep(1)
        if self.isConnected():
            logger.info("Connected to IB")

            self.reqAccountSummary(self.nextOrderId, "All", AccountSummaryTags.AllTags)
        
        else:
            logger.error("Failed to connect to IB")

