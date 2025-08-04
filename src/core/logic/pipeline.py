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
