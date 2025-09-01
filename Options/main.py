import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
from exchange import MarketExchange
from orderbook import Order, Side
from agent import Agent
from noise import generate_daily_price_path #

class Space:
    def __init__(self, low, high, shape):
        self.low = np.array([low])
        self.high = np.array([high])
        self.shape = shape

class TradingEnvironment:
    def __init__(self, start_date, start_cash=1_000_000, active_ticker_count=11):
        self.current_date = start_date
        self.current_step = 0
        self.daily_price_path = generate_daily_price_path(start_date)
        while self.daily_price_path is None :
            print(f"Skipping non-trading day: {self.current_date}")
            self.current_date += timedelta(days=1)
            self.daily_price_path = generate_daily_price_path(self.current_date)
        
        initial_price = self.daily_price_path[0]
        self.market = MarketExchange(underlying_price=initial_price)
        
        self.active_ticker_count = active_ticker_count
        self.active_tickers = []
        self.initial_cash = start_cash
        self.cash_balance = start_cash
        # self.portfolio = {ticker[0]: 0 for ticker in self.market.tickers}
        
        self.portfolio = defaultdict(int)

        self.state_dimensions = 1 + (self.active_ticker_count * 5)
        self.n_tickers = self.active_ticker_count
        self.n_actions = self.n_tickers + 1
        self.action_space = Space(low=-1.0, high=1.0, shape=(self.n_actions,))

        self._update_active_tickers()

    def reset(self, new_date):
        self.current_date = new_date
        self.current_step = 0
        self.daily_price_path = generate_daily_price_path(new_date)
        while self.daily_price_path is None :
            print(f"Skipping non-trading day: {new_date}")
            new_date += timedelta(days=1)
            self.daily_price_path = generate_daily_price_path(new_date)
        

        opening_price = self.daily_price_path[0]
        self.market = MarketExchange(underlying_price=opening_price)
        self._update_active_tickers()
        return self.get_state()

    def step(self, raw_action):
        self.current_step += 1
        if self.current_step >= len(self.daily_price_path):
            return self.get_state(), 0, True

        new_underlying_price = self.daily_price_path[0]
        self.market.update_market(new_underlying_price)
        self._update_active_tickers()

        old_portfolio_value = self.calculate_portfolio_value()
        
        
        ticker_probabilities = raw_action[:self.n_tickers]
        size_and_direction = raw_action[self.n_tickers]
        chosen_ticker_index = np.argmax(ticker_probabilities)
        ticker_to_trade = self.active_tickers[chosen_ticker_index]
        side = Side.BUY if size_and_direction > 0 else Side.SELL
        requested_size = 10 
        
        trade_size = requested_size
        if side == Side.SELL:
            current_holding = self.portfolio.get(ticker_to_trade, 0)
            trade_size = min(requested_size, current_holding)
        
       
       
        if trade_size > 0:
            target_book = self.market.get_book(ticker_to_trade)
            if target_book:
                bids = target_book.get_bids()
                asks = target_book.get_asks()
                price = -1
                if side == Side.BUY and asks: price = asks[0][0]
                elif side == Side.SELL and bids: price = bids[0][0]

                if price != -1:
                    order = Order(side=side, price=price, size=trade_size, owner_id='AGENT_01')
                    executed_trades = target_book.add_order(order)
                    for trade in executed_trades:
                        cost = trade.price * trade.size
                        if trade.taker_side == Side.BUY:
                            self.cash_balance -= cost
                            self.portfolio[trade.ticker_id] += trade.size
                        else:
                            self.cash_balance += cost
                            self.portfolio[trade.ticker_id] -= trade.size
        
        

        new_portfolio_value = self.calculate_portfolio_value()
        reward = new_portfolio_value - old_portfolio_value-new_portfolio_value*0.003
        next_state = self.get_state()
        
        end_of_day = self.current_step >= len(self.daily_price_path) - 1
        is_bankrupt = new_portfolio_value < (self.initial_cash * 0.5)
        done = end_of_day or is_bankrupt

        return next_state, reward, done
    
    def get_state(self):
        state = np.full(self.state_dimensions, -1.0)
        state[0] = self.cash_balance
        pointer = 1
        for ticker_name in self.active_tickers:
            position_size = self.portfolio.get(ticker_name, 0)
            book = self.market.get_book(ticker_name)
            if book:
                bids = book.get_bids(agent_type='retail'); asks = book.get_asks(agent_type='retail')
                best_bid_price = bids[0][0] if bids else -1.0; best_bid_size = bids[0][1] if bids else -1.0
                best_ask_price = asks[0][0] if asks else -1.0; best_ask_size = asks[0][1] if asks else -1.0
            else:
                best_bid_price, best_bid_size, best_ask_price, best_ask_size = [-1.0]*4
            
            state[pointer:pointer+5] = [position_size, best_bid_price, best_bid_size, best_ask_price, best_ask_size]
            pointer += 5
        return state

    def calculate_portfolio_value(self):
        value = self.cash_balance
        for ticker_name, quantity in self.portfolio.items():
            if quantity > 0:
                book = self.market.get_book(ticker_name)
                if book:
                    bids = book.get_bids(agent_type='institutional')
                    if bids:
                        value += quantity * bids[0][0]
        return value

    def _update_active_tickers(self):
        underlying_price = self.market.underlying_price
        all_options = [t for t in self.market.tickers if t[0] != 'STOCK_UNDERLYING']
        all_options.sort(key=lambda t: abs(t[1] - underlying_price))
        closest_options = [t[0] for t in all_options[:self.active_ticker_count - 1]]
        self.active_tickers = ['STOCK_UNDERLYING'] + closest_options

if __name__ == '__main__':
    start_date = date(2023, 2, 1)
    n_episodes = 5
    
    env = TradingEnvironment(start_date)
    
    agent = Agent(input_dims=[env.state_dimensions], env=env,
                  n_actions=env.n_actions, n_tickers=env.n_tickers)
    
    score_history = []
    current_date = start_date
    
    for i in range(n_episodes):
        observation = env.reset(current_date)
        
        while observation is None:
            print(f"Skipping non-trading day: {current_date}")
            current_date += timedelta(days=1)
            observation = env.reset(current_date)
            
        done = False
        score = 0
        
        print(f" Episode {i+1} for Date: {env.current_date} ")
        
        while not done:
            action = agent.choose_action(observation, evaluate=False)
            observation_, reward, done = env.step(action.numpy()) 
            
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            
            observation = observation_
            score += reward
            
        score_history.append(score)
        print(f"  Episode {i+1} finished. Score: {score:.2f}")
        print(f"  Final Portfolio Value: {env.calculate_portfolio_value():.2f}")
        
        current_date += timedelta(days=1)