from datetime import date, timedelta
from market.exchange import MarketExchange
from Agents.DDPG.tradingenvDDPG import initialize_ddpg_agent, run_ddpg_step
from Agents.SAC.env_sac import initialize_sac_agent, run_sac_step
from market.noise import generate_daily_price_path
import matplotlib.pyplot as plt
import time

class LivePlotter:
    def __init__(self, agent_ids, title='Live Agent Performance', total_steps=375):
        self.agent_ids = agent_ids
        # Create a history list for each agent and the market
        self.history = {agent_id: [] for agent_id in agent_ids}
        self.history['market'] = []
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.ax.set_xlim(0, total_steps)
        
        self.lines = {}
        # Create a line for the market
        self.lines['market'], = self.ax.plot([], [], label='Market Price', linestyle='--')
        # Create a line for each agent
        for agent_id in self.agent_ids:
            self.lines[agent_id], = self.ax.plot([], [], label=agent_id)
            
        self.ax.set_title(title)
        self.ax.set_xlabel("Time Steps")
        self.ax.set_ylabel("Value / Profit")
        self.ax.legend()
        self.ax.grid(True)
        
    def update(self, market_price, agent_portfolios, delay=0.01):
        self.history['market'].append(market_price)
        for agent_id, value in agent_portfolios.items():
            self.history[agent_id].append(value)
        
        # Update all lines with new data
        self.lines['market'].set_data(range(len(self.history['market'])), self.history['market'])
        for agent_id in self.agent_ids:
            self.lines[agent_id].set_data(range(len(self.history[agent_id])), self.history[agent_id])

        self.ax.relim()
        self.ax.autoscale_view(scalex=False) # Keep x-axis fixed
        
        self.fig.canvas.flush_events()
        plt.pause(delay)
if __name__ == '__main__':
    # --- 1. Configuration ---
    start_date = date(2023, 2, 1)
    n_episodes = 5
    
    # --- 2. Setup Agents and Environments using an Assembly Line ---
    traders = []
    # Create DDPG Agents
    for i in range(1, 4): # Creates DDPG_1, DDPG_2, DDPG_3
        agent_id = f"DDPG_{i}"
        agent, env = initialize_ddpg_agent(agent_id=agent_id)
        traders.append((agent, env))

    # Create SAC Market Maker Agents
    for i in range(1, 6): # Creates MM_1 through MM_5
        agent_id = f"MM_{i}"
        agent, env = initialize_sac_agent(agent_id=agent_id)
        traders.append((agent, env))

    # --- 3. Setup the Plotter ---
    agent_ids = [agent.agent_id for agent, env in traders]
    total_steps = 375 * n_episodes
    live_plot = LivePlotter(agent_ids=agent_ids, total_steps=total_steps)
    
    # --- 4. The Main Simulation Loop ---
    current_date = start_date
    for i in range(n_episodes):
        # --- Daily Setup ---
        daily_price_path = generate_daily_price_path(current_date)
        while daily_price_path is None or daily_price_path.size == 0:
            print(f"Skipping non-trading day: {current_date}")
            current_date += timedelta(days=1)
            daily_price_path = generate_daily_price_path(current_date)
            
        # Correctly get the opening price
        opening_price = daily_price_path[0]
        central_market = MarketExchange(underlying_price=opening_price)
        
        print(f"\n--- Starting Episode {i+1} for Date: {current_date} ---")
        
        # --- Inner Loop for the Trading Day ---
        for step in range(len(daily_price_path) - 1):
            is_done = (step == len(daily_price_path) - 2)
            
            # The clean "Assembly Line" for agent turns
            for agent, env in traders:
                if "DDPG" in agent.agent_id:
                    run_ddpg_step(agent, env, central_market, is_done)
                else:
                    run_sac_step(agent, env, central_market)

            # --- Market Update ---
            # Correctly get the new price for the next step
            new_price = daily_price_path[step + 1]
            print(f'new_price:{new_price.:2f}')
            central_market.update_market(new_price)
            
            # --- Update the Dashboard ---
            # Correctly build the portfolio dictionary without errors
            current_portfolios = {}
            for agent, env in traders:
                if "DDPG" in agent.agent_id:
                    value = env._calculate_portfolio_value(central_market)
                    profit = value - env.initial_cash
                    current_portfolios[agent.agent_id] = profit
                else: # For SAC Agents
                    value = agent._calculate_portfolio_value(central_market)
                    profit = value - agent.capital
                    current_portfolios[agent.agent_id] = profit
            
            live_plot.update(new_price, current_portfolios, delay=0.01)
            
        print(f"  Episode {i+1} finished.")
        current_date += timedelta(days=1)
        
    plt.ioff()
    plt.show()