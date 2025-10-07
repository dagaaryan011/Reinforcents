# main.py

import torch
import numpy as np
from tqdm import tqdm
from datetime import date, timedelta

# --- 1. Import Market and Utility Components ---
from src.market.exchange import MarketExchange
from src.market.noise import generate_daily_price_path
from src.market.orderbook import Side

# --- 2. Import All Agent Types and Their Environments ---

# Institutional (DDPG) Agents
from src.agents.insti.agent_insti import Agent_Insti
from src.agents.insti.hybrid_environment import HybridAgentEnvironment
from src.agents.insti.model_setup_insti import LongTermModel

# Market Maker (MM) Agents
# (Assuming filenames from your structure, aliased to avoid name conflicts)
from src.agents.Marketmaker.agent import initialize_MM_agent
from src.agents.Marketmaker.environment import Env as MMEnv, run as run_mm_step

# Retail Agents
# (Aliased to avoid name conflicts with other agents/environments)
from src.agents.retail.agent_retail import Agent as RetailAgent
from src.agents.retail.agent_retail_env import AgentEnvironment as RetailEnv


if __name__ == "__main__":
    # --- A. SIMULATION CONFIGURATION ---
    print("--- Configuring Simulation ---")
    
    # Agent Counts
    N_INSTI_AGENTS = 5
    N_MM_AGENTS = 2
    N_RETAIL_AGENTS = 10
    
    # Time and Episode Parameters
    n_episodes = 300
    option_cycle_days = 2
    start_date = date(2023, 1, 1)
    decision_frequency = 30

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- B. SHARED & PRE-LOADED COMPONENTS ---
    
    # Institutional agents share a single pre-trained RNN for context
    print("--- Loading Shared RNN Context Model for Institutional Agents... ---")
    insti_rnn_model = LongTermModel().to(device)
    # insti_rnn_model.load_state_dict(torch.load('path/to/your/insti_rnn_model.pth'))
    insti_rnn_model.eval()

    # Placeholders for agents and environments
    insti_agents, insti_envs = [], []
    mm_agents, retail_agents, retail_envs = [], [], []
    mm_env = MMEnv() # Market Makers share a single environment object

    # --- C. MAIN SIMULATION LOOP (Date-Driven) ---
    print(f"\n--- Starting simulation for {n_episodes} days... ---")
    current_date = start_date
    days_to_expiry = option_cycle_days

    for i in range(n_episodes):
        # --- Daily Setup: Find next valid trading day ---
        daily_price_path= generate_daily_price_path(current_date)
        while daily_price_path is None or daily_price_path.size == 0:
            current_date += timedelta(days=1)
            daily_price_path = generate_daily_price_path(current_date)
        
        opening_price = daily_price_path[0]

        # --- Cycle Setup: Handle option expiry and re-initialization ---
        if days_to_expiry == option_cycle_days:
            print(f"\n--- New {option_cycle_days}-Day Option Cycle Starting ---")
            
            # 1. Create a single, shared Market Exchange for this cycle
            exchange = MarketExchange(
                underlying_price=opening_price,
                time_to_expiry=(days_to_expiry / 365.0)
            )
            
            # 2. Initialize Institutional Agents
            insti_agents, insti_envs = [], []
            temp_env = HybridAgentEnvironment(agent=None, exchange=exchange)
            for j in range(N_INSTI_AGENTS):
                agent = Agent_Insti(agent_id=f"Insti_{j}", alpha=0.0001, beta=0.001, 
                                    input_dims=temp_env.total_state_size, tau=0.005, 
                                    n_actions=temp_env.n_tickers_to_observe, 
                                    rnn_context_model=insti_rnn_model)
                insti_agents.append(agent)
                insti_envs.append(HybridAgentEnvironment(agent=agent, exchange=exchange))

            # 3. Initialize Market Maker Agents
            mm_agents = []
            mm_env.exchange = exchange
            for j in range(N_MM_AGENTS):
                agent = initialize_MM_agent(agent_id=f"MM_{j}")
                agent.load_models() # Make sure to specify model paths inside this function
                agent.broker.env = mm_env
                mm_agents.append(agent)
            
            # 4. Initialize Retail Agents
            retail_agents, retail_envs = [], []
            for j in range(N_RETAIL_AGENTS):
                # TODO: Update path to your retail agent model
                model_path = fr"D:\NetworkPrediction\data\models\retail\model_rnn_2.pth"
                agent = RetailAgent(agent_id=f"Retail_{j}", model_path=model_path)
                retail_agents.append(agent)
                retail_envs.append(RetailEnv(agent=agent, exchange=exchange))

        else:
            # For other days, just update the existing market
            exchange.set_time_to_expiry(days_to_expiry / 365.0)
            exchange.update_market(opening_price)
        
        print(f"--- Day {i+1}/{n_episodes} | Date: {current_date} | Expiry in: {days_to_expiry} days ---")
        
        # --- D. INTRADAY SIMULATION LOOP (Step-Driven) ---
        for step, price in enumerate(tqdm(daily_price_path, desc=f"Day {i+1} Progress")):
            # First, update the market price for all to see
            exchange.update_market(price)

            # Then, all agents preprocess the new price data
            for agent in insti_agents: agent.preprocess_input(price)
            for env in retail_envs: env.update_state(price)
            # (Assuming MM agents handle their state internally during their run step)
            mm_env.run(days_to_expiry / 365.0, price)
            for agent in mm_agents:
                run_mm_step(agent, mm_env)
            # At each decision point, agents act in their specified order
            if (step + 1) % decision_frequency == 0:
                # 1. INSTITUTIONAL AGENTS ACT
                for agent, env in zip(insti_agents, insti_envs):
                    current_observation = env.get_state()
                    action = agent.choose_action(current_observation)
                    new_observation, reward, done = env.step(action)
                    agent.remember(current_observation, action, reward, new_observation, done)
                    agent.learn()
                # 3. RETAIL AGENTS ACT
                for env in retail_envs:
                    env.make_decision()
        
        # --- E. END-OF-DAY REPORTING ---
        print(f"\n--- Day {i+1} Finished | Closing Price: {daily_price_path[-1]:.2f} ---")
        print("  --- Institutional Agent Portfolios ---")
        for env in insti_envs:
            print(f"    {env.agent.agent_id}: Value: {env.portfolio_value:,.2f}, Cash: {env.cash_balance:,.2f}, Pos: {dict(env.portfolio)}")
        
        print("  --- Market Maker Portfolios ---")
        # (Assuming MM agent has `portfolio_value` and `cash_balance` attributes)
        for agent in mm_agents:
            print("  --- Market Maker Portfolios ---"); [print(f"    {agent.agent_id}: Value: {agent.broker.portfolio_value:,.2f}, Cash: {agent.broker.capital:,.2f}, Pos: {dict(agent.broker.portfolio)}") for agent in mm_agents]
            
        print("  --- Retail Agent Portfolios ---")
        for env in retail_envs:
            print(f"    {env.agent.agent_id}: Cash: {env.cash_balance:,.2f}, Pos: {dict(env.portfolio)}")
        
        # --- F. END-OF-DAY UPDATES ---
        current_date += timedelta(days=1)
        days_to_expiry -= 1

        # --- G. EXPIRY AND SETTLEMENT ---
        if days_to_expiry == 0:
            print("\n--- OPTION EXPIRY & SETTLEMENT ---")
            final_price = daily_price_path[-1]
            for env in insti_envs: env.settle_at_expiry(final_price)
            for env in retail_envs: env.settle_at_expiry(final_price)
            for agent in mm_agents: agent.action_at_expiry(opening_price, final_price) # Based on MM example
            
            days_to_expiry = option_cycle_days
            
    print("\n--- Simulation Complete ---")