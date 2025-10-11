import random
from collections import defaultdict
from enum import Enum
import time

class Side(Enum):
    BUY = 1
    SELL = -1

class Trade:
    def __init__(self, taker_id, maker_id, price, size, ticker_id, timestamp):
        self.taker_id = taker_id
        self.maker_id = maker_id
        self.price = price
        self.size = size
        self.ticker_id = ticker_id
        self.timestamp = timestamp
        self.taker_side = None
        self.maker_side = None

    def __repr__(self):
        return f"Trade({self.ticker_id}, P:{self.price}, S:{self.size})"

class Order:
    def __init__(self, side, price, size, owner_id):
        self.side = side
        self.price = price
        self.size = size
        self.owner_id = owner_id
        self.timestamp = time.time()
        self.order_id = None

class OrderBook:
    def __init__(self, ticker_id, ledger, market_price=0.0):
        self.ticker_id = ticker_id
        self.bids = defaultdict(list)
        self.offers = defaultdict(list)
        self.ledger = ledger
        self.market_price = market_price
        self.notifications = defaultdict(list)
        self._order_id_counter = 0

    def collect_notifications_for(self, agent_id):
        messages = self.notifications.pop(agent_id, [])
        return messages

    def add_order(self, order):
        executed_trades = []
        if not self._is_price_acceptable(order.price):
            return executed_trades
        
        order.order_id = self._new_order_id()
        executed_trades = self._process_order(order)
        return executed_trades

    def get_bids(self, view='full'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
        # --- FIX #2: Filter out any levels with zero size ---
        valid_items = [item for item in summary.items() if item[1] > 0]
        sorted_bids = sorted(valid_items, key=lambda x: x[0], reverse=True)
        return sorted_bids[:5] if view == 'retail' else sorted_bids

    def get_asks(self, view='full'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
        # --- FIX #2: Filter out any levels with zero size ---
        valid_items = [item for item in summary.items() if item[1] > 0]
        sorted_asks = sorted(valid_items, key=lambda x: x[0])
        return sorted_asks[:5] if view == 'retail' else sorted_asks

    def _prune_stale_orders(self):
        for price in list(self.bids.keys()):
            if not self._is_price_acceptable(price):
                del self.bids[price]
        for price in list(self.offers.keys()):
            if not self._is_price_acceptable(price):
                del self.offers[price]

    def _new_order_id(self):
        self._order_id_counter += 1
        return self._order_id_counter

    def _is_price_acceptable(self, price, price_limit_pct=0.20):
        if self.market_price is None or self.market_price <= 0: return False
        lower_bound = self.market_price * (1 - price_limit_pct)
        upper_bound = self.market_price * (1 + price_limit_pct)
        return lower_bound <= price <= upper_bound

    def _maintain_book_depth(self, depth=5, max_size=50):
        #In case market mei liquidity nahi aa rahi , Activate this in exchange.py(psuedo market makers)
        if self.market_price is None or not isinstance(self.market_price, (int, float)) or self.market_price <= 0:
            return
        for i in range(1, depth + 1):
            price = round(self.market_price - (i * 0.05), 2)
            if price > 0 and not self.bids.get(price):
                order = Order(Side.BUY, price, random.randint(10, max_size), 'LIQUIDITY_PROVIDER')
                self.bids[order.price].append(order)
        for i in range(1, depth + 1):
            price = round(self.market_price + (i * 0.05), 2)
            if not self.offers.get(price):
                order = Order(Side.SELL, price, random.randint(10, max_size), 'LIQUIDITY_PROVIDER')
                self.offers[order.price].append(order)

    def _process_order(self, incoming_order):
        trades = []
        if incoming_order.side == Side.BUY:
            levels, prices = self.offers, sorted(self.offers.keys())
            while incoming_order.size > 0 and prices and incoming_order.price >= prices[0]:
                trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
                if not levels.get(prices[0]):
                    prices.pop(0)
        else: # SELL
            levels, prices = self.bids, sorted(self.bids.keys(), reverse=True)
            while incoming_order.size > 0 and prices and incoming_order.price <= prices[0]:
                trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
                if not levels.get(prices[0]):
                    prices.pop(0)

        if incoming_order.size > 0:
            levels[incoming_order.price].append(incoming_order)
        return trades

    def _match_at_level(self, incoming_order, levels, price):
        trades = []
        orders_at_level = levels[price]
        for i, book_order in enumerate(orders_at_level):
            if incoming_order.size == 0: break
            trade_size = min(incoming_order.size, book_order.size)
            new_trade = Trade(incoming_order.owner_id, book_order.owner_id, price, trade_size, self.ticker_id, time.time())
            new_trade.taker_side = incoming_order.side
            new_trade.maker_side = book_order.side
            trades.append(new_trade)
            self.ledger.record_trade(new_trade)
            
            incoming_order.size -= trade_size
            book_order.size -= trade_size
            
            maker_id = book_order.owner_id
            
            self.notifications[maker_id].append(new_trade)
        
        levels[price] = [o for o in orders_at_level if o.size > 0]
        return trades