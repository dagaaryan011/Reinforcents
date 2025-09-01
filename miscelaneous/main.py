# main.py

import numpy as np
from orderbook import OrderBook, Order, Side  # Your market simulation
from agent import Agent          # Your DDPG agent
# Assuming your noise function is in a file named `noise.py`
from noise import generate_daily_price_path
from datetime import date,timedelta, datetime

class Space:
    def __init__(self, low, high, shape):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = shape


class TradingEnvironment:
    def __init__(self,current_date):
        initial_price_path = generate_daily_price_path(current_date)

        starting_price = initial_price_path[0]

        self.order_book = OrderBook(market_price=starting_price)
        
        
        self.n_actions = 1
        self.action_space = Space(low=[-1.0], high=[1.0], shape=(self.n_actions,))
        self.input_dims = (4,) # [best_bid, best_ask, shares_held, cash_balance]
        
        self.initial_cash = 3500000
        self.cash_balance = self.initial_cash
        self.shares_held_atstart = 100
        self.shares_held = self.shares_held_atstart
        
        self.price_path = []
        self.current_step = 0

    def get_state(self):
        bids = self.order_book.get_bids()
        asks = self.order_book.get_asks()
        best_bid = bids[0][0] if bids else self.order_book.market_price
        best_ask = asks[0][0] if asks else self.order_book.market_price
        state = np.array([best_bid, best_ask, self.shares_held, self.cash_balance])
        return state
    # observation = env.reset(env.shares_held,env.cash_balance,current_date)
    def reset(self,shares,cash,date1):
        self.price_path = generate_daily_price_path(date1)
        self.current_step = 0
        
        # Reset 
        self.order_book.bids.clear()
        self.order_book.offers.clear()
        self.order_book.market_price = self.price_path[self.current_step]
        stock = shares
        self.cash_balance = cash
        self.shares_held = stock
        
        self.order_book._maintain_book_depth()
        self.current_step=0
        return self.get_state()
    # In the TradingEnvironment class

    # Change the arguments of the reset method
    # def reset(self, daily_price_path):
    #     self.price_path = daily_price_path # Use the path passed from the main loop
    #     self.current_step = 0
        
    #     # Reset the market
    #     self.order_book.bids.clear()
    #     self.order_book.offers.clear()
    #     self.order_book.market_price = self.price_path[self.current_step]

    #     # Reset the portfolio (using the new starting price)
    #     starting_price = self.price_path[0]
    #     self.shares_held = 100 # Or your desired starting shares
    #     self.cash_balance = self.initial_cash - (self.shares_held * starting_price)
        
    #     self.order_book._maintain_book_depth()
    #     return self.get_state()
    def step(self, action):
        # --- 3. ADVANCE TIME (Added) ---
        current_price = self.price_path[self.current_step]
        self.order_book.market_price = current_price
        self.current_step += 1
        
        # --- Take a snapshot of portfolio value BEFORE the action ---
        bids = self.order_book.get_bids()
        asks = self.order_book.get_asks()
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else self.order_book.market_price
        old_portfolio_value = self.cash_balance + (self.shares_held * mid_price)
        
        # --- Translate agent's action into a trade ---
        action_value = action.numpy()[0] 
        max_trade_size = int((self.cash_balance/mid_price))//2
        trade_size = int(abs(action_value) * max_trade_size)

        if trade_size > 0:
            if action_value > 0: # Agent wants to BUY
                if asks:
                    price = asks[0][0]
                    cost = price * trade_size
                    if self.cash_balance >= cost:
                        buy_order = Order(side=Side.BUY, price=price, size=trade_size)
                        self.order_book.add_order(buy_order)
                        self.shares_held += trade_size
                        self.cash_balance -= cost
            elif action_value < 0: # Agent wants to SELL
                if bids and self.shares_held >= trade_size:
                    price = bids[0][0]
                    sell_order = Order(side=Side.SELL, price=price, size=trade_size)
                    self.order_book.add_order(sell_order)
                    self.shares_held -= trade_size
                    self.cash_balance += price * trade_size
        
        # --- Take a snapshot of portfolio value AFTER the action ---
        bids = self.order_book.get_bids()
        asks = self.order_book.get_asks()
        new_mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else self.order_book.market_price
        new_portfolio_value = self.cash_balance + (self.shares_held * new_mid_price)

        # --- Calculate reward and get next state ---
        reward = new_portfolio_value - old_portfolio_value - 50000
        next_state = self.get_state()
        
        # --- 4. ADD "END OF DAY" CONDITION (Modified) ---
        bankrupt = new_portfolio_value < (self.initial_cash * 0.5)
        end_of_day = self.current_step >= len(self.price_path) - 1
        done = bankrupt or end_of_day
        
        return next_state, reward, done

# --- Main execution block ---

import matplotlib.pyplot as plt

def plot_learning_curve(scores, filename='ddpg_learning_curve.png'):
    """
    Plots the scores over episodes and saves the figure.
    """
    # Create an array for the x-axis (episodes)
    x = [i+1 for i in range(len(scores))]
    
    # Calculate a running average of the last 100 scores to see the trend
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        
    plt.figure(figsize=(12, 6))
    plt.plot(x, running_avg)
    plt.title('Running Average of Previous 10 Scores')
    plt.xlabel('Episode')
    plt.ylabel('Average Score (Profit/Loss)')
    plt.grid(True)
    plt.savefig(filename) # Save the plot to a file
    plt.show() # Display the plot


if __name__ == '__main__':
    current_date = date(2023, 2, 1)
    # start_date = datetime.strptime( '2024-07-25', '%Y-%m-%d')
    # current_date = start_date

    env = TradingEnvironment(current_date)    
    agent = Agent(input_dims=env.input_dims,
                  env=env, # Agent needs env to know action bounds
                  n_actions=env.n_actions)
    
    n_episodes = 5
    observation = env.reset(env.shares_held,env.cash_balance,current_date)
    print("--- Setup Complete ---")
    print(f"State space dimensions: {env.input_dims}")
    print(f"Number of actions: {env.n_actions}")
    print(f"Action space low: {env.action_space.low}, high: {env.action_space.high}")
    
    score_history = []
    
    for i in range(n_episodes):
        
        
        daily_price_path = generate_daily_price_path(current_date)
        while daily_price_path is None:
            print(f"Skipping non-trading day: {current_date}")
            current_date += timedelta(days=1) # Move to the next day
            daily_price_path = generate_daily_price_path(current_date)
        
        observation = env.reset(env.shares_held,env.cash_balance,current_date)    
        startingstockprice = daily_price_path[0]
        print (f'Portfolio at the start of day = {env.shares_held*startingstockprice + env.cash_balance:.2f}')
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            
            observation_, reward, done = env.step(action)
            
            
            # 5. Agent stores the experience (state, action, reward, next_state, done)
            agent.remember(observation, action, reward, observation_, done)
            
            agent.learn()
            
            # 7. Update the current state and the total score for the episode
            observation = observation_
            score += reward
        

        score_history.append(score)
        
        # Print the results for this episode
        print(f'Episode {i+1} for date {current_date}')
        print(f'Shares held {env.shares_held}, Trading capital {env.cash_balance:.2f}')
        closing_price = daily_price_path[-1]
        
        print(f"Day's Closing Price: {closing_price:.2f}")
        print (f'Portfolio at the end of day = {env.shares_held*closing_price + env.cash_balance:.2f}')
        current_date += timedelta(days=1)
        
        
        # print(f'current share price {env.current_price}')
    #    ...
    print("\n--- Training Samaptam ---")
    Initial_price = startingstockprice
    initial_porfolio = 3500000 + 100*Initial_price 
    final_price = env.order_book.market_price 
    final_portfolio_value = env.cash_balance + (env.shares_held * final_price)
    print(f"Initial Portfolio -> {initial_porfolio}")
    print(f"Final Portfolio -> Cash: {env.cash_balance:.2f}, Shares: {env.shares_held}")
    print(f"Final Portfolio Value: {final_portfolio_value:.2f}")
    print (f'income-> {final_portfolio_value - initial_porfolio :.2f}')
    plot_learning_curve(score_history)