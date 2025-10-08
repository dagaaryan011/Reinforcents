import numpy as np
from datetime import date, timedelta
import matplotlib.pyplot as plt
import time

# Import custom modules for the exchange, agents, and price generation.
from Market.exchange import MarketExchange
from Agents.MarketMaker.environment import Env, run
from Agents.MarketMaker.agent import initialize_MM_agent
from Agents.RetailTrader.environment import Envn, runn
from Agents.RetailTrader.agent import initialize_RT_agent
from Market.noise import generate_daily_price_path

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
    n_episodes = 300  # Each episode represents one trading day.
    option_time = 10  # the total time of options trading
    # --- Agent Initialization ---

    MM_env = Env()
    RT_env = Envn()

    mm_traders = []
    for i in range(1,2):
        mm_traders.append(initialize_MM_agent(agent_id=f"MM_{i}"))
    for agent in mm_traders:
        agent.load_models()

    rt_traders = []
    for i in range(1,2):
        rt_traders.append(initialize_RT_agent(agent_id=f"RT_{i}"))
    for agent in rt_traders:
        agent.load_models()


        
    # print(type(ddpg_traders[0][0]), type(ddpg_traders[0][1]))
    # print(type(mm_traders[0][0]), type(mm_traders[0][1]))

    # Initialize five SAC agents (to act as market makers).
    # for i in range(1, 6):
    

    # Prepare for plotting.
    # ddpg_agent_ids = [agent.agent_id for agent, env in ddpg_traders]
    # mm_agent_ids = [agent.agent_id for agent, env in mm_traders]
    # agent_ids = ddpg_agent_ids + mm_agent_ids
    # total_steps = 5 * n_episodes
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
            central_market = MarketExchange( time_to_expiry = days_to_expiry / 365, underlying_price=opening_price)
            initial_price = opening_price
        else:
            central_market.get_time_to_expiry(days_to_expiry / 365)
            central_market.update_market(opening_price)

        
        tickers = [t[0] for t in central_market.tickers if t[0] != 'STOCK_UNDERLYING']
        print(tickers)

        MM_env.exchange = central_market
        RT_env.exchange = central_market
        for agent in mm_traders:
            agent.broker.env = MM_env
        for agent in rt_traders:
            agent.broker.env = RT_env
        

        print(f"\n--- Starting Episode {i+1} for Date: {current_date} ---")
        
        # --- Intraday Simulation Loop (by minute/step) ---
        for step in range(len(daily_price_path) - 1):

            # Flag to indicate the end of the day.
            is_done = (step == len(daily_price_path) - 2)
            
            # for agent, env in mm_traders:
            #     run_sac_step(agent, daily_price_path[step], agent.agent_id, days_to_expiry)
            MM_env.run(days_to_expiry/365, daily_price_path[step])
            for agent in mm_traders:
                run(agent, MM_env)
            RT_env.run(days_to_expiry/365, daily_price_path[step])
            for agent in rt_traders:
                run(agent, RT_env)

            # Each agent takes an action in t
            # for agent, env in ddpg_traders:
            #     run_ddpg_step(agent, env, central_market, is_done)
                # else:
                #     run_sac_step(agent, env, central_market)

            # Update the market to the next price in the path.
            new_price = daily_price_path[step + 1]
            # central_market.update_market(new_price)

            # Calculate and plot the current portfolio value for each agent.
            # current_portfolios = {}
            # for agent, env in ddpg_traders:
            #     if "DDPG" in agent.agent_id:
            #         port_value = env._calculate_portfolio_value(central_market)
            #         profit = port_value - env.initial_cash
            #         current_portfolios[agent.agent_id] = profit
                # else: # For SAC agents
                #     value = agent._calculate_portfolio_value(central_market)
                #     profit = value - agent.capital
                #     current_portfolios[agent.agent_id] = profit
            
                #live_plot.update(new_price, current_portfolios, delay=0.01)

        
        print(f"  Episode {i+1} finished.")
        MM_env.indicator()
        RT_env.indicator()
        
        # Move to the next day.
        current_date += timedelta(days=1)
        days_to_expiry -= 1


        if (days_to_expiry == 0):
            print("it is expiry")
            #send the signal here , geting everybody settledup
            final_price = daily_price_path[-1]
            for agent in mm_traders:
                agent.action_at_expiry(initial_price, final_price)
            MM_env.action_at_expiry()

            for agent in rt_traders:
                agent.action_at_expiry(initial_price, final_price)
            RT_env.action_at_expiry()


            # for agent, env in traders:
            #     if "DDPG" in agent.agent_id:
            #         env.settle_expired_positions(final_price)
            #     if "MM" in agent.agent_id:
            #         pass
            
            days_to_expiry = option_time
            pass
        
            with open("C:\ProjectX\OptionsTrading\Market\master_trades.csv", 'w') as f:
                        pass
    
        
    # Keep the final plot window open after the simulation finishes.
    # plt.ioff()
    # plt.show()

