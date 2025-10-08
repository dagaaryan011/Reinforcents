from ...market.orderbook import Order, Side
from config import LOT_SIZE
from collections import defaultdict
import numpy as np
#TODO change the limit of the amount an option can be bought
class Broker:
    def __init__(self):
        
        self.start_cash = 100000
        self.temp_capital = 100000
        self.capital = 100000

        self.start_amount = 100
        self.temp_inventory = 100
        self.inventory = 100
        self.portfolio_value = 0
        self.portfolio = defaultdict(int)
        self.cash_settlement = {}

        self.env = None    # one common connected to the market
    def _calculate_portfolio_value(self):
        """
        Calculates the total net worth of the agent.
        Value = Cash + (Stock Inventory * Stock Price) + (Value of Option Positions)
        """
        # 1. Start with the current cash balance
        value = self.capital

        # 2. Add the value of the underlying stock inventory
        # (Assuming self.inventory is the number of shares of the underlying stock)
        stock_price = self.env.exchange.underlying_price
        value += self.inventory * stock_price

        # 3. Add the value of all open option positions
        for ticker, quantity in self.portfolio.items():
            if quantity == 0:
                continue
            
            book = self.env.exchange.get_book(ticker)
            # Use the best bid to get a conservative "mark-to-market" price
            if book and book.get_bids():
                market_price = book.get_bids()[0][0]
                value += quantity * market_price * LOT_SIZE
        
        # Store the calculated value and return it
        self.portfolio_value = value
        return value
    def get_actual_state(self,ticker):
        if self.portfolio[ticker]<=10 and self.portfolio[ticker]>= -10:
            state = self.env.get_state(ticker)
            state.append(self.capital)
            state.append(self.inventory)
            state.append(self.portfolio[ticker])
        else:
            zero = np.zeros(14).tolist()
            state = zero
        return state
    
    def get_all_states(self):
        states = []
        for ticker in self.env.tickers_list:
            states.append(self.get_actual_state(ticker))

        return states
    
    def get_notifications(self, agent_id):
        #TODO change the function for proper execution
        for ticker in self.env.tickers_list: 
            ob = self.env.exchange.get_book(ticker)
            notifs = ob.collect_notifications_for(agent_id)
            for notif in notifs:
                if notif.maker_side == Side.BUY:
                    self.portfolio[notif.ticker_id] += notif.size
                    self.inventory += notif.size
                    
                    self.capital -= notif.price * notif.size
                    
                    #print(notif.size)
                if notif.maker_side == Side.SELL:
                    self.portfolio[notif.ticker_id] -= notif.size
                    self.inventory -= notif.size
                    
                    self.capital += notif.price * notif.size
                    
                    #print(notif.size)
    
    # def update_book(self, ticker, b_p, a_p, b_s, a_s, agent_id):     # TODO change logic here , will put the order , if execute , change potfolio
    #     buy_order = Order(Side.BUY, b_p, b_s, owner_id=agent_id)
    #     sell_order = Order(Side.SELL, a_p, a_s, owner_id=agent_id)
    #     ob = self.env.exchange.get_book(ticker)
    #     executed_trades_buy = ob.add_order(buy_order)
    #     executed_trades_sell = ob.add_order(sell_order)

    #     done_buy_price = sum(t.price*t.size for t in executed_trades_buy)
    #     done_buy_size = sum(t.size for t in executed_trades_buy)
    #     done_sell_price = sum(t.price*t.size for t in executed_trades_sell)
    #     done_sell_size = sum(t.size for t in executed_trades_sell)

    #     remaining_buy_size = b_s - done_buy_size
    #     remaining_sell_size = a_s - done_sell_size

    #     self.capital += done_sell_price - done_buy_price 
    #     self.temp_capital += done_sell_price - done_buy_price

    #     self.inventory += done_buy_size - done_sell_size
    #     self.temp_inventory += done_buy_size - done_sell_size
        
    #     self.temp_capital -= b_p * remaining_buy_size
    #     self.temp_inventory -= remaining_sell_size
    #     self.portfolio[ticker] += done_buy_size - done_sell_size

    #     # print(b_p, a_p, b_s, a_s)
    #     # print(self.capital)
    #     # print(self.temp_capital)
    #     # print(self.inventory)
    #     # print(self.temp_inventory)
    #     # print(self.portfolio[ticker])

    #     # return done_buy_price, done_sell_price, done_buy_size, done_sell_size
    def update_book(self, ticker, b_p, a_p, b_s, a_s, agent_id):
        """Place buy/sell orders and handle executions"""
        # Place buy order
        buy_order = Order(Side.BUY, b_p, b_s, agent_id)
        buy_trades = self.env.exchange.get_book(ticker).add_order(buy_order)

        # Place sell order  
        sell_order = Order(Side.SELL, a_p, a_s, agent_id)
        sell_trades = self.env.exchange.get_book(ticker).add_order(sell_order)

        # Process buy executions
        for trade in buy_trades:
            self.portfolio[ticker] += trade.size
            self.capital -= trade.price * trade.size * LOT_SIZE

        # Process sell executions  
        for trade in sell_trades:
            self.portfolio[ticker] -= trade.size
            self.capital += trade.price * trade.size * LOT_SIZE

        print(f'buy_filled: {len(buy_trades)}')
        print(f'sell_filled: {len(sell_trades)}')
        # Return simple summary
        # return {
        #     'buy_filled': len(buy_trades),
        #     'sell_filled': len(sell_trades)
        # }
        
            
    def reset_portfolio(self):
        self.portfolio.clear()
        
    def settlement(self, final_price: float):
        """Correct settlement logic for options"""
        for ticker, amount in self.portfolio.items():
            # if amount == 0:
            #     continue

            try:
                # Extract strike price and option type safely
                parts = ticker.split('_')
                if len(parts) < 3:
                    continue

                strike_price = int(parts[1])
                option_type = parts[2]  # 'CE' or 'PE'

                settlement_value = 0

                if amount > 0:  # LONG positions (we own options)
                    if option_type == 'CE':  # Long Call
                        settlement_value = max(0, final_price - strike_price)
                    elif option_type == 'PE':  # Long Put
                        settlement_value = max(0, strike_price - final_price)
                    # Long positions GET money
                    self.cash_settlement[ticker] = settlement_value * amount * LOT_SIZE

                elif amount < 0:  # SHORT positions (we sold options)
                    if option_type == 'CE':  # Short Call
                        settlement_value = max(0, final_price - strike_price)
                    elif option_type == 'PE':  # Short Put  
                        settlement_value = max(0, strike_price - final_price)
                    # Short positions PAY money (negative cash flow)
                    self.cash_settlement[ticker] = -settlement_value * abs(amount) * LOT_SIZE
                
                else:
                    self.cash_settlement[ticker] = 0
            except (ValueError, IndexError):
                print(f"Warning: Could not parse ticker {ticker}")
                continue

    def settle(self):   #settlement action called at end of expiry
        total_settlement = 0
        for ticker in self.env.tickers_list:
            total_settlement += self.cash_settlement[ticker]
        self.capital += total_settlement
        print("total settlement ", total_settlement)


    def get_PL(self, initial, final):
        PL = ( self.capital  ) - ( self.start_cash )
        self.start_cash = self.capital
        return PL
        

    def new_day(self):
        self.temp_capital = self.capital
        self.temp_inventory = self.inventory

 
