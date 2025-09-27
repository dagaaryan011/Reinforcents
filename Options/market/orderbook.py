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
        self.ticker_id = ticker_id      # Which instrument was traded.
        self.price = price              # The price at which the trade occurred.
        self.size = size                # The quantity traded.
        self.taker_id = taker_order.owner_id      # ID of the agent who initiated the trade.
        self.taker_side = taker_order.side        # BUY/SELL side of the initiator.
        self.taker_order_id = taker_order.order_id
        self.maker_id = maker_order.owner_id      # ID of the agent whose resting order was hit.
        self.maker_side = maker_order.side        # BUY/SELL side of the passive party.
        self.maker_order_id = maker_order.order_id
        self.timestamp = time.time()    # When the trade happened.

class Order:
    def __init__(self, side, price, size, owner_id='NOBODY', order_id=None):
        self.side = side                # BUY or SELL.
        self.price = price              # The specified price of the order.
        self.size = size                # The quantity of the order.
        self.owner_id = owner_id        # Which agent placed the order.
        self.order_id = order_id        # A unique ID for the order.
        self.timestamp = time.time()    # When the order was placed.

# Manages all buy (bids) and sell (offers) orders for a single ticker.
class OrderBook:
    def __init__(self, ticker_id, market_price, ledger):
        self.ticker_id = ticker_id      # The instrument this book is for (e.g., 'STOCK_2800_CE').
        self.bids = defaultdict(list)   # A dictionary to store buy orders, keyed by price.
        self.offers = defaultdict(list)# A dictionary to store sell orders, keyed by price.
        self.order_id = 0               # A counter to generate unique order IDs.
        self.market_price = market_price # The current theoretical price, used for validation.
        self.ledger = ledger            # Reference to the master trade ledger.
        self.notifications = defaultdict(list) # A "mailbox" to notify agents whose resting orders are filled.
        min_depth = 10
        self._maintain_book_depth()
       
    # A method for an agent to check for and collect its trade notifications.
    def collect_notifications_for(self, agent_id):
        if agent_id in self.notifications:
            messages = self.notifications[agent_id]
            del self.notifications[agent_id] # Clear the mailbox after collection.
            return messages
        return []

    # The main entry point for placing a new order into the book.
    def add_order(self, order):
        executed_trades = []
        # Reject the order if its price is too far from the current market price.
        if not self._is_price_acceptable(order.price):
            return executed_trades
        
        # Assign a new ID to the order.
        order.order_id = self._new_order_id()
        # Try to match the order against existing orders in the book.
        executed_trades = self._process_order(order)
        
        return executed_trades

    # Returns the current state of the buy side of the book.
    def get_bids(self, agent_type='retail'):
        # Summarize total size available at each price level.
        summary = {price: sum(o.size for o in orders) for price, orders in self.bids.items()}
        # Sort bids from highest price to lowest.
        full_sorted_bids = sorted(summary.items(), key=lambda x: x[0], reverse=True)
        # Retail agents only see the top 5 levels; institutional see the full book.
        return full_sorted_bids[:5] if agent_type == 'retail' else full_sorted_bids

    # Returns the current state of the sell side of the book.
    def get_asks(self, agent_type='retail'):
        summary = {price: sum(o.size for o in orders) for price, orders in self.offers.items()}
        # Sort asks from lowest price to highest.
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

     
    def _process_order(self, incoming_order):
        trades = []
        # If it's a BUY order, try to match it against the SELL side (offers).
        if incoming_order.side == Side.BUY:
            levels = self.offers
            # Get all sell price levels, sorted from lowest to highest.
            prices = sorted(levels.keys())
            # Keep matching as long as the buyer's price is good enough and they still want to buy.
            while incoming_order.size > 0 and prices and incoming_order.price >= prices[0]:
                trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
                # If a price level is cleared, remove it.
                if not levels.get(prices[0]):
                    prices.pop(0)
        # If it's a SELL order, try to match it against the BUY side (bids).
        else:
            levels = self.bids
            # Get all buy price levels, sorted from highest to lowest.
            prices = sorted(levels.keys(), reverse=True)
            while incoming_order.size > 0 and prices and incoming_order.price <= prices[0]:
                trades.extend(self._match_at_level(incoming_order, levels, prices[0]))
                if not levels.get(prices[0]):
                    prices.pop(0)
        
        # If any part of the order remains unfilled, add it to the book as a resting order.
        if incoming_order.size > 0:
            book_side = self.bids if incoming_order.side == Side.BUY else self.offers
            book_side[incoming_order.price].append(incoming_order)
        return trades
            
    # Matches an incoming order against all resting orders at a specific price level.
    def _match_at_level(self, incoming_order, levels, price):
        trades_at_level = []
        orders_at_level = levels[price]
        # Loop through a copy of the orders at this level.
        for book_order in orders_at_level[:]:
            if incoming_order.size == 0: break # Stop if the incoming order is fully filled.
            
            # The trade size is the smaller of the two orders.
            trade_size = min(incoming_order.size, book_order.size)
            # Create a Trade object to record the transaction.
            new_trade = Trade(self.ticker_id, book_order.price, trade_size, incoming_order, book_order)
            self.ledger.record_trade(new_trade)
            trades_at_level.append(new_trade)
            
            maker_id = book_order.owner_id
            if maker_id not in ['NOBODY', 'LIQUIDITY_PROVIDER']:
                self.notifications[maker_id].append(new_trade)
            
            incoming_order.size -= trade_size
            book_order.size -= trade_size
        
        # Update the list of orders at this price level, removing any that are fully filled.
        levels[price] = [o for o in orders_at_level if o.size > 0]
        # If the level is now empty, delete it.
        if not levels[price]:
            del levels[price]
        return trades_at_level
    
    def _maintain_book_depth(self, min_depth=10, price_spread_pct=0.05, max_size=50):
        num_bids = len(self.bids)
        num_offers = len(self.offers)
    
        bids_to_add = min_depth - num_bids
        offers_to_add = min_depth - num_offers
    
        if bids_to_add > 0:
            for _ in range(bids_to_add):
                price = self._round_to_tick(random.uniform(
                    self.market_price * (1 - price_spread_pct), self.market_price * 0.995))
                size = random.randint(1, max_size)
                order = Order(side=Side.BUY, price=price, size=size, 
                              owner_id='LIQUIDITY_PROVIDER')
                self.bids[order.price].append(order)


        if offers_to_add > 0:
            for _ in range(offers_to_add):
                price = self._round_to_tick(random.uniform(
                    self.market_price * 1.005, self.market_price * (1 + price_spread_pct)))
                size = random.randint(1, max_size)
                order = Order(side=Side.SELL, price=price, size=size, 
                              owner_id='LIQUIDITY_PROVIDER')
                self.offers[order.price].append(order)

    def _round_to_tick(self, price, tick_size=0.05):
        return round(round(price / tick_size) * tick_size, 2)
