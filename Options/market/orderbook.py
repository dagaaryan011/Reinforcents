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
#         self.notifications = defaultdict(list) # NEW: The mailbox for maker notifications
#         self._maintain_book_depth()

#     # NEW: A public method for the environment to check an agent's mail
#     def collect_notifications_for(self, agent_id):
#         if agent_id in self.notifications:
#             messages = self.notifications[agent_id]
#             del self.notifications[agent_id]
#             return messages
#         return []

#     def add_order(self, order):
#         executed_trades = []
#         order.price = self._round_to_tick(order.price)
#         if not self._is_price_acceptable(order.price):
#             return executed_trades
        
#         order.order_id = self._new_order_id()
#         executed_trades = self._process_order(order)
#         self._maintain_book_depth()
#         return executed_trades

#     def get_bids(self, agent_type='retail'):
#         summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
#         full_sorted_bids = sorted(summary.items(), key=lambda x: x[0], reverse=True)
#         return full_sorted_bids[:5] if agent_type == 'retail' else full_sorted_bids

#     def get_asks(self, agent_type='retail'):
#         summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
#         full_sorted_asks = sorted(summary.items(), key=lambda x: x[0])
#         return full_sorted_asks[:5] if agent_type == 'retail' else full_sorted_asks

#     def _round_to_tick(self, price, tick_size=0.05):
#         return round(round(price / tick_size) * tick_size, 2)

#     def _new_order_id(self):
#         self.order_id += 1
#         return self.order_id

    
#     def remove_invalid_orders(self):
#         for price in list(self.bids.keys()):
#             if not self._is_price_acceptable(price):
#                 del self.bids[price]
#         for price in list(self.offers.keys()):
#             if not self._is_price_acceptable(price):
#                 del self.offers[price]

                
#     def _is_price_acceptable(self, price, price_limit_pct=0.20):
#         lower_bound = self.market_price * (1 - price_limit_pct)
#         upper_bound = self.market_price * (1 + price_limit_pct)
#         return lower_bound <= price <= upper_bound

#     def _maintain_book_depth(self, min_depth=10, price_spread_pct=0.05, max_size=50):
#         num_bids = len(self.bids)
#         num_offers = len(self.offers)
    
#         bids_to_add = min_depth - num_bids
#         offers_to_add = min_depth - num_offers
    
#         if bids_to_add > 0:
#             for _ in range(bids_to_add):
#                 price = self._round_to_tick(random.uniform(
#                     self.market_price * (1 - price_spread_pct), self.market_price * 0.995))
#                 size = random.randint(1, max_size)
#                 order = Order(side=Side.BUY, price=price, size=size, 
#                               owner_id='LIQUIDITY_PROVIDER')
#                 self.bids[order.price].append(order)

#         if offers_to_add > 0:
#             for _ in range(offers_to_add):
#                 price = self._round_to_tick(random.uniform(
#                     self.market_price * 1.005, self.market_price * (1 + price_spread_pct)))
#                 size = random.randint(1, max_size)
#                 order = Order(side=Side.SELL, price=price, size=size, 
#                               owner_id='LIQUIDITY_PROVIDER')
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
#             self.ledger.record_trade(new_trade)
#             trades_at_level.append(new_trade)
            
#             # NEW: Notify the maker by putting the trade in their mailbox
#             maker_id = book_order.owner_id
#             if maker_id != 'NOBODY':
#                 self.notifications[maker_id].append(new_trade)
            
#             incoming_order.size -= trade_size
#             book_order.size -= trade_size
        
#         levels[price] = [o for o in orders_at_level if o.size > 0]
#         if not levels[price]:
#             del levels[price]
#         return trades_at_level




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
    def __init__(self, ticker_id, price, size, taker_order, maker_order):
        self.ticker_id = ticker_id
        self.price = price
        self.size = size
        self.taker_id = taker_order.owner_id
        self.taker_side = taker_order.side
        self.taker_order_id = taker_order.order_id
        self.maker_id = maker_order.owner_id
        self.maker_side = maker_order.side
        self.maker_order_id = maker_order.order_id
        self.timestamp = time.time()

class Order:
    def __init__(self, side, price, size, owner_id='NOBODY', order_id=None):
        self.side = side
        self.price = price
        self.size = size
        self.owner_id = owner_id
        self.order_id = order_id
        self.timestamp = time.time()

class OrderBook:
    def __init__(self, ticker_id, market_price, ledger):
        self.ticker_id = ticker_id
        self.bids = defaultdict(list)
        self.offers = defaultdict(list)
        self.order_id = 0
        self.market_price = market_price
        self.ledger = ledger
        self.notifications = defaultdict(list)
       

    def collect_notifications_for(self, agent_id):
        if agent_id in self.notifications:
            messages = self.notifications[agent_id]
            del self.notifications[agent_id]
            return messages
        return []

    def add_order(self, order):
        executed_trades = []
        if not self._is_price_acceptable(order.price):
            return executed_trades
        print(f"ORDER: {order.owner_id} : ({self.ticker_id}): price:{order.price}: {order.side}")
        order.order_id = self._new_order_id()
        executed_trades = self._process_order(order)
        
        return executed_trades

    def get_bids(self, agent_type='retail'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
        full_sorted_bids = sorted(summary.items(), key=lambda x: x[0], reverse=True)
        return full_sorted_bids[:5] if agent_type == 'retail' else full_sorted_bids

    def get_asks(self, agent_type='retail'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
        full_sorted_asks = sorted(summary.items(), key=lambda x: x[0])
        return full_sorted_asks[:5] if agent_type == 'retail' else full_sorted_asks
        
    def _prune_stale_orders(self):
        for price in list(self.bids.keys()):
            if not self._is_price_acceptable(price):
                del self.bids[price]
        for price in list(self.offers.keys()):
            if not self._is_price_acceptable(price):
                del self.offers[price]

   

    def _new_order_id(self):
        self.order_id += 1
        return self.order_id

    def _is_price_acceptable(self, price, price_limit_pct=0.20):
        lower_bound = self.market_price * (1 - price_limit_pct)
        upper_bound = self.market_price * (1 + price_limit_pct)
        return lower_bound <= price <= upper_bound

    # def _maintain_book_depth(self, min_depth=10, price_spread_pct=0.05, max_size=50):
    #     num_bids = len(self.bids)
    #     num_offers = len(self.offers)
    #     bids_to_add = min_depth - num_bids
    #     offers_to_add = min_depth - num_offers
        
    #     if bids_to_add > 0:
    #         for _ in range(bids_to_add):
    #             price = self.market_price
    #             size = random.randint(1, max_size)
    #             order = Order(side=Side.BUY, price=price, size=size, owner_id='LIQUIDITY_PROVIDER')
    #             self.bids[order.price].append(order)
    #     if offers_to_add > 0:
    #         for _ in range(offers_to_add):
    #             price = self.market_price
    #             size = random.randint(1, max_size)
    #             order = Order(side=Side.SELL, price=price, size=size, owner_id='LIQUIDITY_PROVIDER')
    #             self.offers[order.price].append(order)

    def _process_order(self, incoming_order):
        trades = []
        if incoming_order.side == Side.BUY:
            levels = self.offers
            prices = sorted(levels.keys())
            while incoming_order.size > 0 and prices and incoming_order.price >= prices[0]:
                trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
                if not levels.get(prices[0]):
                    prices.pop(0)
        else:
            levels = self.bids
            prices = sorted(levels.keys(), reverse=True)
            while incoming_order.size > 0 and prices and incoming_order.price <= prices[0]:
                trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
                if not levels.get(prices[0]):
                    prices.pop(0)
        
        if incoming_order.size > 0:
            book_side = self.bids if incoming_order.side == Side.BUY else self.offers
            book_side[incoming_order.price].append(incoming_order)
        return trades
            
    def _match_at_level(self, incoming_order, levels, price):
        trades_at_level = []
        orders_at_level = levels[price]
        for book_order in orders_at_level[:]:
            if incoming_order.size == 0: break
            
            trade_size = min(incoming_order.size, book_order.size)
            new_trade = Trade(self.ticker_id, book_order.price, trade_size, incoming_order, book_order)
            print(f" Trade : {new_trade.taker_id} [{new_trade.taker_side.name}] matched with {new_trade.maker_id} [{new_trade.maker_side.name}] for {new_trade.size} units @ {new_trade.price} in {self.ticker_id}")
            self.ledger.record_trade(new_trade)
            trades_at_level.append(new_trade)
            
            maker_id = book_order.owner_id
            if maker_id != 'NOBODY' and maker_id != 'LIQUIDITY_PROVIDER':
                self.notifications[maker_id].append(new_trade)
            
            incoming_order.size -= trade_size
            book_order.size -= trade_size
        
        levels[price] = [o for o in orders_at_level if o.size > 0]
        if not levels[price]:
            del levels[price]
        return trades_at_level