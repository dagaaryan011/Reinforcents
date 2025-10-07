# # import pandas as pd
# # import enum
# # import time
# # from pathlib import Path
# # from collections import defaultdict
# # import random

# # class Side(enum.Enum):
# #     BUY = "BUY"
# #     SELL = "SELL"

# # class Trade:
# #     def __init__(self, ticker_id, price, size, taker_order, maker_order):
# #         self.ticker_id = ticker_id
# #         self.price = price
# #         self.size = size
# #         self.taker_id = taker_order.owner_id
# #         self.taker_side = taker_order.side
# #         self.taker_order_id = taker_order.order_id
# #         self.maker_id = maker_order.owner_id
# #         self.maker_side = maker_order.side
# #         self.maker_order_id = maker_order.order_id
# #         self.timestamp = time.time()

# # class Order:
# #     def __init__(self, side, price, size, owner_id='NOBODY', order_id=None):
# #         self.side = side
# #         self.price = price
# #         self.size = size
# #         self.owner_id = owner_id
# #         self.order_id = order_id
# #         self.timestamp = time.time()

# # class OrderBook:
# #     def __init__(self, ticker_id, market_price, ledger):
# #         self.ticker_id = ticker_id
# #         self.bids = defaultdict(list)
# #         self.offers = defaultdict(list)
# #         self.order_id = 0
# #         self.market_price = market_price
# #         self.ledger = ledger
# #         self.notifications = defaultdict(list) # NEW: The mailbox for maker notifications
# #         self._maintain_book_depth()

# #     # NEW: A public method for the environment to check an agent's mail
# #     def collect_notifications_for(self, agent_id):
# #         if agent_id in self.notifications:
# #             messages = self.notifications[agent_id]
# #             del self.notifications[agent_id]
# #             return messages
# #         return []

# #     def add_order(self, order):
# #         executed_trades = []
# #         order.price = self._round_to_tick(order.price)
# #         if not self._is_price_acceptable(order.price):
# #             return executed_trades
        
# #         order.order_id = self._new_order_id()
# #         executed_trades = self._process_order(order)
# #         self._maintain_book_depth()
# #         return executed_trades

# #     def get_bids(self, agent_type='retail'):
# #         summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
# #         full_sorted_bids = sorted(summary.items(), key=lambda x: x[0], reverse=True)
# #         return full_sorted_bids[:5] if agent_type == 'retail' else full_sorted_bids

# #     def get_asks(self, agent_type='retail'):
# #         summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
# #         full_sorted_asks = sorted(summary.items(), key=lambda x: x[0])
# #         return full_sorted_asks[:5] if agent_type == 'retail' else full_sorted_asks

# #     def _round_to_tick(self, price, tick_size=0.05):
# #         return round(round(price / tick_size) * tick_size, 2)

# #     def _new_order_id(self):
# #         self.order_id += 1
# #         return self.order_id

    
# #     def remove_invalid_orders(self):
# #         for price in list(self.bids.keys()):
# #             if not self._is_price_acceptable(price):
# #                 del self.bids[price]
# #         for price in list(self.offers.keys()):
# #             if not self._is_price_acceptable(price):
# #                 del self.offers[price]

                
# #     def _is_price_acceptable(self, price, price_limit_pct=0.20):
# #         lower_bound = self.market_price * (1 - price_limit_pct)
# #         upper_bound = self.market_price * (1 + price_limit_pct)
# #         return lower_bound <= price <= upper_bound

# #     def _maintain_book_depth(self, min_depth=10, price_spread_pct=0.05, max_size=50):
# #         num_bids = len(self.bids)
# #         num_offers = len(self.offers)
    
# #         bids_to_add = min_depth - num_bids
# #         offers_to_add = min_depth - num_offers
    
# #         if bids_to_add > 0:
# #             for _ in range(bids_to_add):
# #                 price = self._round_to_tick(random.uniform(
# #                     self.market_price * (1 - price_spread_pct), self.market_price * 0.995))
# #                 size = random.randint(1, max_size)
# #                 order = Order(side=Side.BUY, price=price, size=size, 
# #                               owner_id='LIQUIDITY_PROVIDER')
# #                 self.bids[order.price].append(order)

# #         if offers_to_add > 0:
# #             for _ in range(offers_to_add):
# #                 price = self._round_to_tick(random.uniform(
# #                     self.market_price * 1.005, self.market_price * (1 + price_spread_pct)))
# #                 size = random.randint(1, max_size)
# #                 order = Order(side=Side.SELL, price=price, size=size, 
# #                               owner_id='LIQUIDITY_PROVIDER')
# #                 self.offers[order.price].append(order)

# #     def _process_order(self, incoming_order):
# #         trades = []
# #         if incoming_order.side == Side.BUY:
# #             levels = self.offers
# #             prices = sorted(levels.keys())
# #             while incoming_order.size > 0 and prices and incoming_order.price >= prices[0]:
# #                 trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
# #                 if not levels.get(prices[0]):
# #                     prices.pop(0)
# #         else:
# #             levels = self.bids
# #             prices = sorted(levels.keys(), reverse=True)
# #             while incoming_order.size > 0 and prices and incoming_order.price <= prices[0]:
# #                 trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
# #                 if not levels.get(prices[0]):
# #                     prices.pop(0)
        
# #         if incoming_order.size > 0:
# #             book_side = self.bids if incoming_order.side == Side.BUY else self.offers
# #             book_side[incoming_order.price].append(incoming_order)
# #         return trades
            
# #     def _match_at_level(self, incoming_order, levels, price):
# #         trades_at_level = []
# #         orders_at_level = levels[price]
# #         for book_order in orders_at_level[:]:
# #             if incoming_order.size == 0: break
            
# #             trade_size = min(incoming_order.size, book_order.size)
            
# #             new_trade = Trade(self.ticker_id, book_order.price, trade_size, incoming_order, book_order)
# #             self.ledger.record_trade(new_trade)
# #             trades_at_level.append(new_trade)
            
# #             # NEW: Notify the maker by putting the trade in their mailbox
# #             maker_id = book_order.owner_id
# #             if maker_id != 'NOBODY':
# #                 self.notifications[maker_id].append(new_trade)
            
# #             incoming_order.size -= trade_size
# #             book_order.size -= trade_size
        
# #         levels[price] = [o for o in orders_at_level if o.size > 0]
# #         if not levels[price]:
# #             del levels[price]
# #         return trades_at_level




# import pandas as pd
# import enum
# import time
# from pathlib import Path
# from collections import defaultdict
# import random

# class Side(enum.Enum):
#     BUY = "BUY"
#     SELL = "SELL"

# class Trade:
#     def __init__(self, ticker_id, price, size, taker_order, maker_order):
#         self.ticker_id = ticker_id
#         self.price = price
#         self.size = size
#         self.taker_id = taker_order.owner_id
#         self.taker_side = taker_order.side
#         self.taker_order_id = taker_order.order_id
#         self.maker_id = maker_order.owner_id
#         self.maker_side = maker_order.side
#         self.maker_order_id = maker_order.order_id
#         self.timestamp = time.time()

# class Order:
#     def __init__(self, side, price, size, owner_id='NOBODY', order_id=None):
#         self.side = side
#         self.price = price
#         self.size = size
#         self.owner_id = owner_id
#         self.order_id = order_id
#         self.timestamp = time.time()

# class OrderBook:
#     def __init__(self, ticker_id, market_price, ledger):
#         self.ticker_id = ticker_id
#         self.bids = defaultdict(list)
#         self.offers = defaultdict(list)
#         self.order_id = 0
#         self.market_price = market_price
#         self.ledger = ledger
#         self.notifications = defaultdict(list)
       

#     def collect_notifications_for(self, agent_id):
#         if agent_id in self.notifications:
#             messages = self.notifications[agent_id]
#             del self.notifications[agent_id]
#             return messages
#         return []

#     def add_order(self, order):
#         executed_trades = []
#         if not self._is_price_acceptable(order.price):
#             return executed_trades
#         print(f"ORDER: {order.owner_id} : ({self.ticker_id}): price:{order.price}: {order.side}")
#         order.order_id = self._new_order_id()
#         executed_trades = self._process_order(order)
        
#         return executed_trades

#     def get_bids(self, agent_type='retail'):
#         summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
#         full_sorted_bids = sorted(summary.items(), key=lambda x: x[0], reverse=True)
#         return full_sorted_bids[:5] if agent_type == 'retail' else full_sorted_bids

#     def get_asks(self, agent_type='retail'):
#         summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
#         full_sorted_asks = sorted(summary.items(), key=lambda x: x[0])
#         return full_sorted_asks[:5] if agent_type == 'retail' else full_sorted_asks
        
#     def _prune_stale_orders(self):
#         for price in list(self.bids.keys()):
#             if not self._is_price_acceptable(price):
#                 del self.bids[price]
#         for price in list(self.offers.keys()):
#             if not self._is_price_acceptable(price):
#                 del self.offers[price]

   

#     def _new_order_id(self):
#         self.order_id += 1
#         return self.order_id

#     def _is_price_acceptable(self, price, price_limit_pct=0.20):
#         lower_bound = self.market_price * (1 - price_limit_pct)
#         upper_bound = self.market_price * (1 + price_limit_pct)
#         return lower_bound <= price <= upper_bound

#     # In src/market/orderbook.py

# # Make sure you have this import at the top of the file


# # ... inside the OrderBook class ...
#     # In src/market/orderbook.py

#     def _maintain_book_depth(self, depth=5, max_size=50):
#         """
#         A simple liquidity bot that adds orders to ensure the book is not empty.
#         """
#         # Guard clause to prevent errors if the book's market price is invalid
#         if self.market_price is None or not isinstance(self.market_price, (int, float)) or self.market_price <= 0:
#             return
    
#         # Add Bids (Buy orders)
#         for i in range(1, depth + 1):
#             price = round(self.market_price - (i * 0.05), 2)
#             if price > 0 and not self.bids.get(price):
#                 size = random.randint(10, max_size)
#                 order = Order(side=Side.BUY, price=price, size=size, owner_id='LIQUIDITY_PROVIDER')
#                 self.bids[order.price].append(order)
    
#         # Add Asks (Sell orders)
#         for i in range(1, depth + 1):
#             price = round(self.market_price + (i * 0.05), 2)
#             if not self.offers.get(price):
#                 size = random.randint(10, max_size)
#                 order = Order(side=Side.SELL, price=price, size=size, owner_id='LIQUIDITY_PROVIDER')
#                 self.offers[order.price].append(order)
#     def _process_order(self, incoming_order):
#         trades = []
#         if incoming_order.side == Side.BUY:
#             levels = self.offers
#             prices = sorted(levels.keys())
#             while incoming_order.size > 0 and prices and incoming_order.price >= prices[0]:
#                 trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
#                 if not levels.get(prices[0]):
#                     prices.pop(0)
#         else:
#             levels = self.bids
#             prices = sorted(levels.keys(), reverse=True)
#             while incoming_order.size > 0 and prices and incoming_order.price <= prices[0]:
#                 trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
#                 if not levels.get(prices[0]):
#                     prices.pop(0)
        
#         if incoming_order.size > 0:
#             book_side = self.bids if incoming_order.side == Side.BUY else self.offers
#             book_side[incoming_order.price].append(incoming_order)
#         return trades
            
#     def _match_at_level(self, incoming_order, levels, price):
#         trades_at_level = []
#         orders_at_level = levels[price]
#         for book_order in orders_at_level[:]:
#             if incoming_order.size == 0: break
            
#             trade_size = min(incoming_order.size, book_order.size)
#             new_trade = Trade(self.ticker_id, book_order.price, trade_size, incoming_order, book_order)
#             print(f" Trade : {new_trade.taker_id} [{new_trade.taker_side.name}] matched with {new_trade.maker_id} [{new_trade.maker_side.name}] for {new_trade.size} units @ {new_trade.price} in {self.ticker_id}")
#             self.ledger.record_trade(new_trade)
#             trades_at_level.append(new_trade)
            
#             maker_id = book_order.owner_id
#             if maker_id != 'NOBODY' and maker_id != 'LIQUIDITY_PROVIDER':
#                 self.notifications[maker_id].append(new_trade)
            
#             incoming_order.size -= trade_size
#             book_order.size -= trade_size
        
#         levels[price] = [o for o in orders_at_level if o.size > 0]
#         if not levels[price]:
#             del levels[price]
#         return trades_at_level
# In src/market/orderbook.py
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
            trades.append(new_trade)
            self.ledger.record_trade(new_trade)
            
            incoming_order.size -= trade_size
            book_order.size -= trade_size
            
            maker_id = book_order.owner_id
            if maker_id != 'LIQUIDITY_PROVIDER':
                self.notifications[maker_id].append(new_trade)
        
        levels[price] = [o for o in orders_at_level if o.size > 0]
        return trades