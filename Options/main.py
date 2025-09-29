import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Import custom modules for the exchange, agents, and price generation.
from market.exchange import MarketExchange
from Agents.DDPG.tradingenvDDPG import initialize_ddpg_agent, run_ddpg_step
from Agents.SAC.env_sac import initialize_sac_agent, run_sac_step, action_at_expiry
from market.noise import generate_daily_price_path
from Agents.SAC.networks_sac import spreadActorNetwork, spreadCriticNetwork, spreadValueNetwork


# A class to handle real-time plotting of agent performance and market price.
class LivePlotter:
    def __init__(self, agent_ids, title='Live Agent Performance', total_steps=375):
        self.agent_ids = agent_ids
        # History dictionaries to store data points for plotting.
        self.history = {agent_id: [] for agent_id in agent_ids}
        self.history['market'] = []
        
        # Turn on interactive mode for live plotting.
        plt.ion()
        # Create a figure with two subplots, sharing the x-axis.
        self.fig, self.axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Configure the top plot for agent profits.
        self.axs[0].set_title(title)
        self.axs[0].set_ylabel("Agent Profit")
        self.axs[0].grid(True)
        
        # Configure the bottom plot for the market price.
        self.axs[1].set_ylabel("Market Price")
        self.axs[1].set_xlabel("Time Steps")
        self.axs[1].grid(True)
        
        # Create line objects for each agent and the market.
        self.lines = {}
        for agent_id in self.agent_ids:
            self.lines[agent_id], = self.axs[0].plot([], [], label=agent_id)
        self.lines['market'], = self.axs[1].plot([], [], label='Market Price', color='darkorange', linestyle='--')
        
        # Add legends to both plots.
        self.axs[0].legend()
        self.axs[1].legend()
        
    def update(self, market_price, agent_portfolios, delay=0.01):
        """
        Updates the plot with new data for the market and all agents.
        """
        # Append new data to the history.
        self.history['market'].append(market_price)
        for agent_id, value in agent_portfolios.items():
            self.history[agent_id].append(value)
        
        # Update the data for each line object.
        self.lines['market'].set_data(range(len(self.history['market'])), self.history['market'])
        for agent_id in self.agent_ids:
            self.lines[agent_id].set_data(range(len(self.history[agent_id])), self.history[agent_id])

        # Rescale the axes to fit the new data.
        self.axs[0].relim()
        self.axs[0].autoscale_view()
        self.axs[1].relim()
        self.axs[1].autoscale_view()
        
        # Redraw the plot.
        self.fig.canvas.flush_events()
        plt.pause(delay)

# This block is the main execution part of the script.
if __name__ == '__main__':
    # --- Simulation Parameters ---
    start_date = date(2023, 2, 1)
    n_episodes = 15  # Each episode represents one trading day.
    option_time = 3
    # --- Agent Initialization ---
    ddpg_traders = []
    mm_traders = []
    # Initialize one DDPG agent (the main learner).
    for i in range(1, 31):
        ddpg_traders.append(initialize_ddpg_agent(agent_id=f"DDPG_{i}"))


    for i in range(1, 21):
        mm_traders.append(initialize_sac_agent(agent_id=f"MM_{i}"))

        
    # print(type(ddpg_traders[0][0]), type(ddpg_traders[0][1]))
    # print(type(mm_traders[0][0]), type(mm_traders[0][1]))

    # Initialize five SAC agents (to act as market makers).
    # for i in range(1, 6):
    

    # Prepare for plotting.
    ddpg_agent_ids = [agent.agent_id for agent, env in ddpg_traders]
    mm_agent_ids = [agent.agent_id for agent, env in mm_traders]
    agent_ids = ddpg_agent_ids + mm_agent_ids
    total_steps = 5 * n_episodes
    #live_plot = LivePlotter(agent_ids=agent_ids, total_steps=total_steps)
    
    days_to_expiry = option_time
    current_date = start_date
    daily_price_path = [0]
    for i in range(n_episodes):

        # Generate a price path for the day, skipping non-trading days.
        daily_price_path = generate_daily_price_path(current_date)
        while daily_price_path is None or daily_price_path.size == 0:
            print(f"Skipping non-trading day: {current_date}")
            current_date += timedelta(days=1)
            daily_price_path = generate_daily_price_path(current_date)
            
        # Create a new market for the day with the opening price.
        opening_price = daily_price_path[0]
        if days_to_expiry == option_time:
            central_market = MarketExchange( time = days_to_expiry / 365, underlying_price=opening_price)
            initial_price = opening_price

        
        tickers = [t[0] for t in central_market.tickers if t[0] != 'STOCK_UNDERLYING']
        print(tickers)
        central_market.update_market(daily_price_path[0])

        for k in range(0,20):
            mm_traders[k][0].env.exchange = central_market
            mm_traders[k][0].ticker = tickers[k]

        
        print(f"\n--- Starting Episode {i+1} for Date: {current_date} ---")
        
        # --- Intraday Simulation Loop (by minute/step) ---
        for step in range(len(daily_price_path) - 1):

            # Flag to indicate the end of the day.
            is_done = (step == len(daily_price_path) - 2)
            
            for agent, env in mm_traders:
                run_sac_step(agent, daily_price_path[step], agent.agent_id, days_to_expiry)

            # Each agent takes an action in t
            for agent, env in ddpg_traders:
                run_ddpg_step(agent, env, central_market, is_done)
                # else:
                #     run_sac_step(agent, env, central_market)

            # Update the market to the next price in the path.
            new_price = daily_price_path[step + 1]
            # central_market.update_market(new_price)

            # Calculate and plot the current portfolio value for each agent.
            current_portfolios = {}
            for agent, env in ddpg_traders:
                if "DDPG" in agent.agent_id:
                    port_value = env._calculate_portfolio_value(central_market)
                    profit = port_value - env.initial_cash
                    current_portfolios[agent.agent_id] = profit
                # else: # For SAC agents
                #     value = agent._calculate_portfolio_value(central_market)
                #     profit = value - agent.capital
                #     current_portfolios[agent.agent_id] = profit
            
                #live_plot.update(new_price, current_portfolios, delay=0.01)
        for agent, env in mm_traders:
            agent.env.eod()

        
            
        print(f"  Episode {i+1} finished.")
        
        # Move to the next day.
        current_date += timedelta(days=1)
        days_to_expiry -= 1


        if (days_to_expiry == 0):
            #send the signal here , geting everybody settledup
            final_price = daily_price_path[-1]
            for agent, env in mm_traders:
                action_at_expiry(agent, final_price, initial_price)
                agent.env.reset_portfolio()
                agent.save_weights()
            with open("C:\ProjectX\Options\market\master_trades.csv", 'w') as f:
                        pass


            # for agent, env in traders:
            #     if "DDPG" in agent.agent_id:
            #         env.settle_expired_positions(final_price)
            #     if "MM" in agent.agent_id:
            #         pass
            
            days_to_expiry = option_time
            pass
    
        
    # Keep the final plot window open after the simulation finishes.
    # plt.ioff()
    # plt.show()

