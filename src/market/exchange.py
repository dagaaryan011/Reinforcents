
import random
from collections import defaultdict
from .blackscholes import BlaScho
from .orderbook import OrderBook
from .trade_records import TradeLedger
from ..tools.functions import calculate_historical_volatility

class MarketExchange:
    def __init__(self, underlying_price=1300.0, strike_interval=5, option_depth=6, time_to_expiry=(30/365.0)):
        self.underlying_price = underlying_price
        self.strike_interval = strike_interval
        self.option_depth = option_depth
        self.time_to_expiry = time_to_expiry
        self.price_history = [underlying_price] # For historical vol calc
        
        self.market_books = {}
        self.tickers = []
        self.ledger = TradeLedger()
        
        self.atm = round(self.underlying_price / self.strike_interval) * self.strike_interval
        self._generate_tickers()
        self._create_all_order_books()
        self.update_market(self.underlying_price) # Initial price

    def get_book(self, ticker_name):
        return self.market_books.get(ticker_name)

    def set_time_to_expiry(self, new_time_to_expiry):
        self.time_to_expiry = new_time_to_expiry

    def update_market(self, new_underlying_price):
        self.underlying_price = new_underlying_price
        self.price_history.append(new_underlying_price)
        self.atm = round(self.underlying_price / self.strike_interval) * self.strike_interval
        
        for ticker_tuple in self.tickers:
            book = self.get_book(ticker_tuple[0])
            if book:
                new_book_market_price = self.calculate_initial_premium(ticker_tuple)
                if new_book_market_price is not None:
                    book.market_price = new_book_market_price
                
                # First, add new orders based on the new price
                # book._maintain_book_depth()
                # Then, clean up any very old orders that are now stale
                book._prune_stale_orders()

    def _generate_tickers(self):
        #making the relevant orderbooks for the current underlying price
        #here we will create the diffrent names for the orderbook to easily access them later
        self.tickers.append(('STOCK_UNDERLYING', self.underlying_price))
        for i in range(-self.option_depth, self.option_depth + 1):
            strike = self.atm + (i * self.strike_interval)
            if strike <= 0: continue
            self.tickers.append((f"STOCK_{strike}_CE", strike))
            self.tickers.append((f"STOCK_{strike}_PE", strike))

    def _create_all_order_books(self):
        #creation of the orderbook
        for ticker_name, strike in self.tickers:
            initial_price = self.calculate_initial_premium((ticker_name, strike))
            book = OrderBook(ticker_name, self.ledger, initial_price)
            self.market_books[ticker_name] = book

    def calculate_initial_premium(self, ticker_tuple):
        #A relevant premium for the orderbook so that the price of the asset can be controlled
        
        ticker_name, strike_price = ticker_tuple
        
        if ticker_name == 'STOCK_UNDERLYING':
            return self.underlying_price

        option_type = 'call' if 'CE' in ticker_name else 'put'
        risk_free_rate = 0.05
        volatility = calculate_historical_volatility(self.price_history)
        
        premium = BlaScho(
            spot=self.underlying_price, strike=strike_price, time=self.time_to_expiry,
            ret=risk_free_rate, vol=volatility, opt=option_type
        )
        prem,_,_,_,_ = premium.calculate()
        if  prem > 0:
            return prem   
        else: return 0.013