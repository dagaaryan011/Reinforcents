
import time
from orderbook import OrderBook

class MarketExchange:
    def __init__(self, underlying_price=100, interval=5, strike_depth=6):
        self.underlying_price = underlying_price#market pprice
        self.strike_interval = interval#5rs ka diff 
        self.strike_depth = strike_depth#kitne strike prices tak jana hai
        
        
        self.tickers = []
       
        self.market_books = {}
        
        self.atm = round(self.underlying_price / self.strike_interval) * self.strike_interval
        
        
        self._generate_tickers()
        self._create_all_order_books()

    def _generate_tickers(self):
        tickers = []
        stock_tuple = ('STOCK_UNDERLYING', 0)
        tickers.append(stock_tuple)

        for i in range(-self.strike_depth_depth, self.strike_depth + 1):
            if i == 0:
                continue
            current_strike = self.atm + (i * self.strike_interval)
            if current_strike <= 0:
                continue
            ticker_name_ce = f'STOCK_{int(current_strike)}_CE'
            ticker_name_pe = f'STOCK_{int(current_strike)}_PE'
            tickers.append((ticker_name_ce, int(current_strike)))
            tickers.append((ticker_name_pe, int(current_strike)))
        self.tickers = tickers

   
    def _calculate_initial_premium(self, ticker_tuple):
        ticker_name, strike_price = ticker_tuple
        if ticker_name == "STOCK_UNDERLYING":
            return self.underlying_price
        
        distance_from_money = abs(self.underlying_price - strike_price)
        base_premium = self.underlying_price * 0.05
        premium = max(0.5, base_premium - (distance_from_money * 0.01))
        return round(premium, 2)

    def _create_all_order_books(self):
        for ticker_tuple in self.tickers:
            
            initial_price = self._calculate_initial_premium(ticker_tuple)
            ticker_name = ticker_tuple[0]
            
            new_book = OrderBook(market_price=initial_price, instrument_id=ticker_name)
            
            
            self.market_books[ticker_name] = new_book

    def get_book(self, ticker_name):
      
        book = self.market_books.get(ticker_name)
        if book is None:
            print(f"Warning: No order book for '{ticker_name}'")
        return book