import pandas as pd
import enum
import time
from pathlib import Path
from collections import defaultdict
import random

# --- Define file paths for persistent storage ---
ORDERS_FILE = Path(r"D:\Reinforcents\stockmarket\orders.csv")
TRADES_FILE = Path(r"D:\Reinforcents\stockmarket\trades.csv")


class Side(enum.Enum):
    """Enumeration for order side."""
    BUY = "BUY"
    SELL = "SELL"


class Trade:
    """Represents a single executed trade."""
    def __init__(self, incoming_side, price, size, incoming_order_id, book_order_id):
        self.side = incoming_side
        self.price = price
        self.size = size
        self.incoming_order_id = incoming_order_id
        self.book_order_id = book_order_id

    def __repr__(self):
        return f"Trade(side={self.side.name}, price={self.price}, size={self.size})"


class Order:
    """Represents a single order in the book."""
    def __init__(self, side, price, size, order_id=None):
        self.side = side
        self.price = price
        self.size = size
        self.order_id = order_id
    
    def __str__(self):
        return f"{self.side.name} {self.size}@{self.price:.2f}"
    
    def __repr__(self):
        return f"Order(side={self.side.name}, price={self.price}, size={self.size}, order_id={self.order_id})"


class OrderBook:
    """A complete order book that manages orders, trades, and persistence."""

    def __init__(self, market_price=100.0, price_limit_pct=0.10):
        self.bids = defaultdict(list)
        self.offers = defaultdict(list)
        self.order_id = 0
        self.market_price = market_price
        self.price_limit_pct = price_limit_pct
        self._load_orders_from_csv()
        self._maintain_book_depth()

    # --- Public Methods (for the GUI) ---

    def add_order(self, order):
        """
        Public method to add a new order. It rounds, validates, processes, and saves the state.
        Returns True if the order is accepted, False otherwise.
        """
        order.price = self._round_to_tick(order.price)

        if not self._is_price_acceptable(order.price):
            print(f"Order rejected: Price {order.price} is outside the allowed range.")
            return False
        
        order.order_id = self._new_order_id()
        self._process_order(order)
        self._save_orders_to_csv()
        self._maintain_book_depth()
        return True

    def get_bids(self):
        """Returns a summarized list of bids (price, total_size) for UI display."""
        summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
        return sorted(summary.items(), key=lambda x: x[0], reverse=True)

    def get_asks(self):
        """Returns a summarized list of asks (price, total_size) for UI display."""
        summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
        return sorted(summary.items(), key=lambda x: x[0])

    def get_trades(self):
        """Reads all executed trades from trades.csv for UI display."""
        if not TRADES_FILE.exists() or TRADES_FILE.stat().st_size == 0:
            return []
        try:
            trades_df = pd.read_csv(TRADES_FILE)
            return trades_df[['Side', 'Price', 'Size']].to_records(index=False).tolist()
        except pd.errors.EmptyDataError:
            return []

    # --- Internal Core Logic & Simulation ---

    def _round_to_tick(self, price, tick_size=0.05):
        """Rounds a price to the nearest valid tick size."""
        return round(round(price / tick_size) * tick_size, 2)

    def _new_order_id(self):
        """Generates a new unique order ID."""
        self.order_id += 1
        return self.order_id

    def _is_price_acceptable(self, price):
        """Check if the order price is within the configured percentage of the market price."""
        lower_bound = self.market_price * (1 - self.price_limit_pct)
        upper_bound = self.market_price * (1 + self.price_limit_pct)
        return lower_bound <= price <= upper_bound

    def _maintain_book_depth(self, min_depth=10, price_spread_pct=0.05, max_size=50):
        """Ensures both sides of the book have a minimum number of orders."""
        num_bids = sum(len(orders) for orders in self.bids.values())
        num_offers = sum(len(orders) for orders in self.offers.values())
        
        bids_to_add = min_depth - num_bids
        offers_to_add = min_depth - num_offers
        
        if bids_to_add > 0:
            for _ in range(bids_to_add):
                price = self._round_to_tick(random.uniform(self.market_price * (1 - price_spread_pct), self.market_price * 0.99))
                size = random.randint(1, max_size)
                order = Order(Side.BUY, price, size, order_id=self._new_order_id())
                self.bids[order.price].append(order)

        if offers_to_add > 0:
            for _ in range(offers_to_add):
                price = self._round_to_tick(random.uniform(self.market_price * 1.01, self.market_price * (1 + price_spread_pct)))
                size = random.randint(1, max_size)
                order = Order(Side.SELL, price, size, order_id=self._new_order_id())
                self.offers[order.price].append(order)
        
        if bids_to_add > 0 or offers_to_add > 0:
            self._save_orders_to_csv()

    def _process_order(self, incoming_order):
        """Internal method to process and match a new order against the book."""
        if incoming_order.side == Side.BUY:
            levels = self.offers
            prices = sorted(levels.keys())
            while incoming_order.size > 0 and prices and incoming_order.price >= prices[0]:
                self._match_at_level(incoming_order, levels, prices[0])
                if not levels.get(prices[0]):
                    prices.pop(0)
        else:
            levels = self.bids
            prices = sorted(levels.keys(), reverse=True)
            while incoming_order.size > 0 and prices and incoming_order.price <= prices[0]:
                self._match_at_level(incoming_order, levels, prices[0])
                if not levels.get(prices[0]):
                    prices.pop(0)

        if incoming_order.size > 0:
            book_side = self.bids if incoming_order.side == Side.BUY else self.offers
            book_side[incoming_order.price].append(incoming_order)
            
    def _match_at_level(self, incoming_order, levels, price):
        """Matches an incoming order with resting orders at a specific price level."""
        orders_at_level = levels[price]
        for book_order in orders_at_level[:]:
            if incoming_order.size == 0: break
            trade_size = min(incoming_order.size, book_order.size)
            trade = Trade(incoming_order.side, book_order.price, trade_size, incoming_order.order_id, book_order.order_id)
            self._save_trade_to_csv(trade)
            incoming_order.size -= trade_size
            book_order.size -= trade_size
        
        levels[price] = [o for o in orders_at_level if o.size > 0]
        if not levels[price]:
            del levels[price]

    # --- Internal Persistence Logic (CSV Handling) ---

    def _save_orders_to_csv(self):
        """Saves the current state of open bids and offers to orders.csv."""
        all_orders = [order for level in self.bids.values() for order in level] + \
                     [order for level in self.offers.values() for order in level]
        if not all_orders:
            pd.DataFrame(columns=["ID", "Side", "Price", "Size"]).to_csv(ORDERS_FILE, index=False)
            return
        orders_data = [{"ID": o.order_id, "Side": o.side.value, "Price": o.price, "Size": o.size} for o in all_orders]
        pd.DataFrame(orders_data).to_csv(ORDERS_FILE, index=False)

    def _load_orders_from_csv(self):
        """Initializes the order book from the orders.csv file on startup."""
        if not ORDERS_FILE.exists(): return
        try:
            orders_df = pd.read_csv(ORDERS_FILE)
            if orders_df.empty: return
            for _, row in orders_df.dropna().iterrows():
                order = Order(side=Side[row["Side"]], price=float(row["Price"]), size=int(row["Size"]), order_id=int(row["ID"]))
                if order.side == Side.BUY:
                    self.bids[order.price].append(order)
                else:
                    self.offers[order.price].append(order)
            if not orders_df["ID"].empty:
                self.order_id = int(orders_df["ID"].max())
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"Error loading orders from CSV: {e}")

    def _save_trade_to_csv(self, trade):
        """Appends a single executed trade to the trades.csv file."""
        trade_data = {"Side": trade.side.name, "Price": trade.price, "Size": trade.size,
                      "IncomingOrderID": trade.incoming_order_id, "BookOrderID": trade.book_order_id}
        df = pd.DataFrame([trade_data])
        header = not TRADES_FILE.exists() or TRADES_FILE.stat().st_size == 0
        df.to_csv(TRADES_FILE, mode='a', header=header, index=False)