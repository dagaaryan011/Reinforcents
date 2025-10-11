


import pandas as pd
from pathlib import Path
from config import Trades_CSV_path
class TradeLedger:
    def __init__(self, file_path=Trades_CSV_path):
        self.trades_file = Path(file_path)
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        self.trades = []

    def record_trade(self, trade_object):
        self.trades.append(trade_object)
        self._save_trade_to_csv(trade_object)

    def _save_trade_to_csv(self, trade_object):
        trade_data = {
            "Timestamp": trade_object.timestamp,
            "Ticker": trade_object.ticker_id,
            "Price": trade_object.price,
            "Size": trade_object.size,
            "Side": trade_object.taker_side.name,
        }
        df = pd.DataFrame([trade_data])
        header = not self.trades_file.exists() or self.trades_file.stat().st_size == 0
        df.to_csv(self.trades_file, mode='a', header=header, index=False)



import pandas as pd
from pathlib import Path

class TradeLedger:
    def __init__(self, file_path="C:\ProjectX\OptionsTrading\Market\master_trades.csv"):
        self.trades_file = Path(file_path)
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        self.trades = []

    def record_trade(self, trade_object):
        self.trades.append(trade_object)
        self._save_trade_to_csv(trade_object)

    def _save_trade_to_csv(self, trade_object):
        trade_data = {
            "Timestamp": trade_object.timestamp,
            "Ticker": trade_object.ticker_id,
            "Price": trade_object.price,
            "Size": trade_object.size,
            "Side": trade_object.taker_side.name,
        }
        df = pd.DataFrame([trade_data])
        header = not self.trades_file.exists() or self.trades_file.stat().st_size == 0
        df.to_csv(self.trades_file, mode='a', header=header, index=False)
