import pandas as pd
from pathlib import Path

# Defines a ledger to record and persist all trades.
class TradeLedger:
    """
    Handles the recording of all executed trades and saves them to a CSV file.
    This creates a master log of all market activity.
    """
    def __init__(self, file_path="C:\ProjectX\Options\market\master_trades.csv"):
        # The path to the CSV file where trades will be stored.
        self.trades_file = Path(file_path)
        # Create the directory for the file if it doesn't already exist.
        self.trades_file.parent.mkdir(parents=True, exist_ok=True)
        # An in-memory list to hold trade objects for the current session.
        self.trades = []

    def record_trade(self, trade_object):
        """
        Public method to record a new trade. It adds the trade to the in-memory
        list and triggers the save-to-CSV mechanism.
        """
        self.trades.append(trade_object)
        self._save_trade_to_csv(trade_object)

    def _save_trade_to_csv(self, trade_object):
        """
        Appends a single trade's data to the master CSV file.
        """
        # Create a dictionary containing the relevant data from the trade object.
        trade_data = {
            "Timestamp": trade_object.timestamp,
            "Ticker": trade_object.ticker_id,
            "Price": trade_object.price,
            "Size": trade_object.size,
            "Side": trade_object.taker_side.name, # Records the side of the trade initiator (Taker)
        }
        # Convert the dictionary into a pandas DataFrame.
        df = pd.DataFrame([trade_data])
        # Check if the CSV file needs a header. It does if the file is new or empty.
        header = not self.trades_file.exists() or self.trades_file.stat().st_size == 0
        # Append the DataFrame to the CSV file without the index.
        df.to_csv(self.trades_file, mode='a', header=header, index=False)