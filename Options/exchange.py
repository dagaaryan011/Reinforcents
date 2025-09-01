from orderbook import OrderBook
from trade_records import TradeLedger

class MarketExchange:
    def __init__(self, underlying_price=100, interval=5, option_depth=6):
        self.underlying_price = underlying_price
        self.strike_interval = interval
        self.option_depth = option_depth
        self.tickers = []
        self.market_books = {}
        self.master_ledger = TradeLedger()
        self.atm = round(self.underlying_price / self.strike_interval) * self.strike_interval
        
        self._generate_tickers()
        self._create_all_order_books()

    def _generate_tickers(self):
        tickers = [('STOCK_UNDERLYING', 0)]
        for i in range(-self.option_depth, self.option_depth + 1):
            if i == 0: continue
            current_strike = self.atm + (i * self.strike_interval)
            if current_strike <= 0: continue
            
            tickers.append((f'STOCK_{int(current_strike)}_CE', int(current_strike)))
            tickers.append((f'STOCK_{int(current_strike)}_PE', int(current_strike)))
        self.tickers = tickers

    def calculate_initial_premium(self, ticker_tuple):
        ticker_name, strike_price = ticker_tuple
        if ticker_name == "STOCK_UNDERLYING":
            return self.underlying_price
        
        distance = abs(self.underlying_price - strike_price)
        base_premium = self.underlying_price * 0.05
        premium = max(0.5, base_premium - (distance * 0.1))
        return round(premium, 2)

    def _create_all_order_books(self):
        for ticker_tuple in self.tickers:
            price = self.calculate_initial_premium(ticker_tuple)
            name = ticker_tuple[0]
            
            new_book = OrderBook(ticker_id=name, 
                                 market_price=price, 
                                 ledger=self.master_ledger)
            
            self.market_books[name] = new_book

    def get_book(self, ticker_name):
        return self.market_books.get(ticker_name)
    
    def update_market(self, new_underlying_price):
        self.underlying_price = new_underlying_price
        for ticker_tuple in self.tickers:
            ticker_name = ticker_tuple[0]
            book = self.get_book(ticker_name)
            if book:
                new_book_market_price = self.calculate_initial_premium(ticker_tuple)
                book.market_price = new_book_market_price
                book._prune_stale_orders()
                book._maintain_book_depth()