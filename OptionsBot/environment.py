import numpy as np
from exchange import MarketExchange
import random
import numpy as np
from orderbook import OrderBook, Order, Side
from datetime import datetime, timedelta
from noise2 import path
import matplotlib.pyplot as plt

import numpy as np

class Env:

    def __init__(self):
        self.exchange = MarketExchange(underlying_price=100, interval=5, option_depth=6)
        self.option_ticker_names = [ticker for ticker in self.exchange.market_books if ticker != 'STOCK_UNDERLYING']
        

    def get_orderbook(self):
        random_ticker = random.choice(self.option_ticker_names)
        orderbook = self.exchange.get_book(random_ticker)

        return orderbook
    
    def get_highestbid_lowestask(self, ob, agent_id):
        bids = ob.get_bids(agent_id)
        asks = ob.get_asks(agent_id)

        highest_bid = bids[0][0] if bids else 0 #None
        lowest_ask = asks[0][0] if asks else 0 #None

        return highest_bid, lowest_ask
    
    def update_book(self, ob, b_p, a_p, b_s, a_s, agent_id):

        #print("update")

        sidebuy = Side.BUY
        sidesell = Side.SELL

        pricebuy = b_p
        pricesell = a_p

        sizebuy = b_s
        sizesell = a_s

        ordbuy = Order(sidebuy, pricebuy, sizebuy, owner_id=agent_id)
        ordsell = Order(sidesell, pricesell, sizesell, owner_id=agent_id)

        executed_trades_buy = ob.add_order(ordbuy)
        executed_trades_sell = ob.add_order(ordsell)

        # if executed_trades_buy:
        #     print("Order was matched! Trades executed:")
        #     for trade in executed_trades_buy:
        #         print(f"Traded {trade.size} at {trade.price} with {trade.maker_id}")
        # else:
        #     print("Order added to the book. No immediate execution.")

        # if executed_trades_sell:
        #     print("Order was matched! Trades executed:")
        #     for trade in executed_trades_sell:
        #         print(f"Traded {trade.size} at {trade.price} with {trade.maker_id}")
        # else:
        #     print("Order added to the book. No immediate execution.")
        
        done_buy_price = 0
        done_buy_size = 0
        done_sell_price = 0
        done_sell_size = 0

        if executed_trades_buy:
            for trade in executed_trades_buy:
                done_buy_price += trade.price
                done_buy_size += trade.size

        if executed_trades_buy:
            for trade in executed_trades_sell:
                done_sell_price += trade.price
                done_sell_size += trade.size

        return done_buy_price, done_sell_price, done_buy_size, done_sell_size
    
    def get_reward(self, executed_ask_price, executed_ask_size, executed_bid_price, executed_bid_size, highest_bid, lowest_ask):
        PL = executed_ask_price * executed_ask_size - executed_bid_price * executed_bid_size
        diff_bid = abs(executed_bid_price - highest_bid) 
        diff_ask = abs(lowest_ask - executed_ask_price)
        reward = PL - diff_bid - diff_ask
        size_diff = executed_bid_size - executed_ask_size

        return PL, size_diff, reward

