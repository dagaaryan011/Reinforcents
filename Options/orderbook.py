import pandas as pd
import enum
import time
from pathlib import Path
from collections import defaultdict
import random

class Side(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"

class Trade:
    def __init__(self, incoming_side, price, size, incoming_order_id, book_order_id):
        self.side = incoming_side
        self.price = price
        self.size = size
        self.incoming_order_id = incoming_order_id
        self.book_order_id = book_order_id

    def __repr__(self):
        return f"Trade(side={self.side.name}, price={self.price}, size={self.size})"

class Order:
    def __init__(self, side, price, size, order_id=None):
        self.side = side
        self.price = price
        self.size = size
        self.order_id = order_id
        self.timestamp = time.time()
    
    def __str__(self):
        return f"{self.side.name} {self.size}@{self.price:.2f}"
    
    def __repr__(self):
        return f"Order(side={self.side.name}, price={self.price}, size={self.size}, order_id={self.order_id})"

class OrderBook:
    def __init__(self, ticker_id='DEFAULT', market_price=100.0, price_limit_pct=0.10):
        self.ticker_id = ticker_id
        self.bids = defaultdict(list)
        self.offers = defaultdict(list)
        self.order_id = 0
        self.market_price = market_price
        self.price_limit_pct = price_limit_pct
        
        self.orders_file = Path(f"D:/Reinforcents/stockmarket/data/orders_{self.ticker_id}.csv")
        self.trades_file = Path(f"D:/Reinforcents/stockmarket/data/trades_{self.ticker_id}.csv")
        
        self._load_orders_from_csv()
        self._maintain_book_depth()

    def add_order(self, order):
        order.price = self._round_to_tick(order.price)
        if not self._is_price_acceptable(order.price):
            print(f"Order rejected: Price {order.price} is outside the allowed range for {self.ticker_id}.")
            return False
        
        order.order_id = self._new_order_id()
        self._process_order(order)
        self._save_orders_to_csv()
        self._maintain_book_depth()
        return True

    def get_bids(self, agent_type='retail'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
        full_sorted_bids = sorted(summary.items(), key=lambda x: x[0], reverse=True)
        if agent_type == 'institutional':
            return full_sorted_bids
        else:
            return full_sorted_bids[:5]

    def get_asks(self, agent_type='retail'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
        full_sorted_asks = sorted(summary.items(), key=lambda x: x[0])
        if agent_type == 'institutional':
            return full_sorted_asks
        else:
            return full_sorted_asks[:5]

    def get_trades(self):
        if not self.trades_file.exists() or self.trades_file.stat().st_size == 0:
            return []
        try:
            trades_df = pd.read_csv(self.trades_file)
            return trades_df[['Side', 'Price', 'Size']].to_records(index=False).tolist()
        except pd.errors.EmptyDataError:
            return []

    def _round_to_tick(self, price, tick_size=0.05):
        return round(round(price / tick_size) * tick_size, 2)

    def _new_order_id(self):
        self.order_id += 1
        return self.order_id

    def _is_price_acceptable(self, price):
        lower_bound = self.market_price * (1 - self.price_limit_pct)
        upper_bound = self.market_price * (1 + self.price_limit_pct)
        return lower_bound <= price <= upper_bound

    def _maintain_book_depth(self, min_depth=10, price_spread_pct=0.05, max_size=50):
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

    def _save_orders_to_csv(self):
        all_orders = [order for level in self.bids.values() for order in level] + \
                     [order for level in self.offers.values() for order in level]
        if not all_orders:
            if self.orders_file.exists(): self.orders_file.unlink()
            return
        orders_data = [{"ID": o.order_id, "Side": o.side.value, "Price": o.price, "Size": o.size, "Timestamp": o.timestamp} for o in all_orders]
        pd.DataFrame(orders_data).to_csv(self.orders_file, index=False)

    def _load_orders_from_csv(self):
        if not self.orders_file.exists(): return
        try:
            orders_df = pd.read_csv(self.orders_file)
            if orders_df.empty: return
            for _, row in orders_df.dropna().iterrows():
                order = Order(side=Side[row["Side"]], price=float(row["Price"]), size=int(row["Size"]), order_id=int(row["ID"]))
                if "Timestamp" in row: order.timestamp = float(row["Timestamp"])
                if order.side == Side.BUY:
                    self.bids[order.price].append(order)
                else:
                    self.offers[order.price].append(order)
            if not orders_df["ID"].empty:
                self.order_id = int(orders_df["ID"].max())
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"Error loading orders from {self.orders_file}: {e}")

    def _save_trade_to_csv(self, trade):
        trade_data = {"Side": trade.side.name, "Price": trade.price, "Size": trade.size, "IncomingOrderID": trade.incoming_order_id, "BookOrderID": trade.book_order_id}
        df = pd.DataFrame([trade_data])
        header = not self.trades_file.exists() or self.trades_file.stat().st_size == 0
        df.to_csv(self.trades_file, mode='a', header=header, index=False)