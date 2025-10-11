from ...market.orderbook import Order, Side
from config import LOT_SIZE
from collections import defaultdict
import numpy as np
import random
class Broker:
    def __init__(self):
        
        self.start_cash = 100000
        self.temp_capital = 100000
        self.capital = 100000

        self.start_amount = 100
        self.temp_inventory = 100
        self.inventory = 100
        self.portfolio_value = self.capital
        self.portfolio = defaultdict(int)
        self.cash_settlement = {}

        self.env = None    # one common connected to the market
    # 
    def _calculate_portfolio_value(self):
        """
        Calculates the total value of the agent's portfolio.
        Value = Cash + Marked-to-Market Value of all option positions.
        """
        # Start with the current cash balance.
        value = self.capital
    
        # Loop through all option positions to get their current market value.
        for ticker, quantity in self.portfolio.items():
            if quantity == 0:
                continue
            
            book = self.env.exchange.get_book(ticker)
            if not book:
                continue
            
            price = 0
            # For LONG positions, value them at the current BID price (what you can sell for).
            if quantity > 0 and book.get_bids():
                price = book.get_bids()[0][0]
            # For SHORT positions, value them at the current ASK price (what it costs to buy back).
            elif quantity < 0 and book.get_asks():
                price = book.get_asks()[0][0]
            else:
                # If there's no price, the position can't be valued at this moment.
                continue
            
            # Add the market value of the option position to the total value.
            value += quantity * price * LOT_SIZE
    
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
        """
        Processes notifications for passively filled (maker) orders and correctly
        updates the agent's capital and portfolio.
        """
        for ticker in self.env.tickers_list: 
            book = self.env.exchange.get_book(ticker)
            if not book:
                continue

            notifications = book.collect_notifications_for(agent_id)
            for trade in notifications:
                # --- START: FIX ---

                # Since this is a notification, our agent was the MAKER.
                print(f"    CONFIRMED (Passive): Agent {agent_id} Maker trade executed: {trade}")
                cost_or_revenue = trade.price * trade.size * LOT_SIZE # Correctly calculate full value

                if trade.maker_side == Side.BUY:
                    # Our resting BUY order was filled, so we BOUGHT
                    print(f"      ACTION (Maker): Our resting BUY order filled: +{trade.size} {trade.ticker_id}")
                    self.portfolio[trade.ticker_id] += trade.size
                    self.capital -= cost_or_revenue # Apply the full value

                elif trade.maker_side == Side.SELL:
                    # Our resting SELL order was filled, so we SOLD
                    print(f"      ACTION (Maker): Our resting SELL order filled: -{trade.size} {trade.ticker_id}")
                    self.portfolio[trade.ticker_id] -= trade.size
                    self.capital += cost_or_revenue                
                        
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

        # print(f'buy_filled: {len(buy_trades)}')
        # print(f'sell_filled: {len(sell_trades)}')
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

 
