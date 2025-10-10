
import torch
import numpy as np
from tqdm import tqdm
from datetime import date, timedelta
from src.tools.dashboard_updater import update_data, clear_data

# ---  Market  ---
from src.market.exchange import MarketExchange
from src.market.noise import generate_daily_price_path
from src.market.orderbook import Side
from src.tools.async_creator import async_batch_creator
# --- 2 Agent  ---

# Institutional (DDPG) Agents
from src.agents.insti.agent_insti import Agent_Insti
from src.agents.insti.hybrid_environment import HybridAgentEnvironment
from src.agents.insti.model_setup_insti import LongTermModel

# Market Maker (MM) Agents
from src.agents.Marketmaker.agent import initialize_MM_agent
from src.agents.Marketmaker.environment import Env as MMEnv, run as run_mm_step

# Retail Agents
from src.agents.retail.agent_retail import Agent as RetailAgent
from src.agents.retail.agent_retail_env import AgentEnvironment as RetailEnv


if __name__ == "__main__":
    print("--- Configuring Simulation ---")
    clear_data()
    # Agent Counts
    N_INSTI_AGENTS = 10
    N_MM_AGENTS = 100
    N_RETAIL_AGENTS = 190
    
    # Time and Episode Parameters
    n_episodes = 300
    option_cycle_days = 5
    start_date = date(2023, 1, 1)
    decision_frequency = 30

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    print("--- Loading Shared RNN Context Model for Institutional Agents... ---")
insti_rnn_model = LongTermModel().to(device)
insti_rnn_model.eval()

# Initialize ALL agents ONCE at the start
print("--- Initializing All Agents (One-Time Setup) ---")

# Create initial exchange for agent setup
initial_exchange = MarketExchange(underlying_price=1000.0)  # Temporary price


def calculate_retail_portfolio_value(env, current_price):
    stock_value = sum(quantity * current_price for ticker, quantity in env.portfolio.items())
    return env.cash_balance + stock_value

# 1. Initialize Institutional Agents ONCE

insti_agents, insti_envs = [], []
temp_env = HybridAgentEnvironment(agent=None, exchange=initial_exchange)
for j in range(N_INSTI_AGENTS):
    agent = Agent_Insti(agent_id=f"Insti_{j}", alpha=0.0001, beta=0.001, 
                        input_dims=temp_env.total_state_size, tau=0.005, 
                        n_actions=temp_env.n_tickers_to_observe, 
                        rnn_context_model=insti_rnn_model)
    insti_agents.append(agent)
    # Environments will be recreated with actual exchange later
print("--- Loading Previous DDPG Models (if any) ---")
for agent in insti_agents:
    agent.load_models(0)
# 2. Initialize Market Maker Agents ONCE  
mm_agents = []
mm_env = MMEnv()
for j in range(N_MM_AGENTS):
    agent = initialize_MM_agent(agent_id=f"MM_{j}")
    agent.load_models()
    mm_agents.append(agent)

# 3. Initialize Retail Agents ONCE
retail_agents, retail_envs = [], []
for j in range(N_RETAIL_AGENTS):
    agent = RetailAgent(agent_id=f"Retail_{j}")
    retail_agents.append(agent)
    # Environments will be recreated with actual exchange later

# --- C. MAIN SIMULATION LOOP (Date-Driven) ---
print(f"\n--- Starting simulation for {n_episodes} days... ---")
current_date = start_date
days_to_expiry = option_cycle_days

# Create first exchange
exchange = MarketExchange(
    
    time_to_expiry=(days_to_expiry / 365.0)
)

# Connect agents to the actual exchange
insti_envs = [HybridAgentEnvironment(agent=agent, exchange=exchange) for agent in insti_agents]
retail_envs = [RetailEnv(agent=agent, exchange=exchange) for agent in retail_agents]
mm_env.exchange = exchange
for agent in mm_agents:
    agent.broker.env = mm_env

for i in range(n_episodes):
    # --- Daily Setup: Find next valid trading day ---
    daily_price_path = generate_daily_price_path(current_date)
    while daily_price_path is None or daily_price_path.size == 0:
        current_date += timedelta(days=1)
        daily_price_path = generate_daily_price_path(current_date)
    
    opening_price = daily_price_path[0]
    exchange = MarketExchange(
    underlying_price=opening_price,
    time_to_expiry=(days_to_expiry / 365.0)
                                )

    if days_to_expiry == option_cycle_days:
        print(f"\n--- New {option_cycle_days}-Day Option Cycle Starting ---")
        
        # JUST update the existing exchange with new parameters
        exchange = MarketExchange(
            underlying_price=opening_price,
            time_to_expiry=(days_to_expiry / 365.0)
        )
        
        # Update environments with the new exchange (KEEP existing agents!)
        for env in insti_envs:
            env.exchange = exchange
        for env in retail_envs:
            env.exchange = exchange
        mm_env.exchange = exchange
        
    else:
        # For other days, just update the existing market
        exchange.set_time_to_expiry(days_to_expiry / 365.0)
        exchange.update_market(opening_price)
    
    print(f"--- Day {i+1}/{n_episodes} | Date: {current_date} | Expiry in: {days_to_expiry} days ---")
    batch1, batch2, batch3 = async_batch_creator(insti_agents, retail_agents, mm_agents)
    all_batches = [batch1, batch2, batch3]

    # Create a mapping from agent ID to its environment for easy lookup
    insti_env_map = {env.agent.agent_id: env for env in insti_envs}
    retail_env_map = {env.agent.agent_id: env for env in retail_envs}
    # --- D. INTRADAY SIMULATION LOOP (Step-Driven) ---
    for step, price in enumerate(tqdm(daily_price_path, desc=f"Day {i+1} Progress")):
        
        exchange.update_market(price)
        for agent in insti_agents: agent.preprocess_input(price)
        for env in retail_envs: env.update_state(price)
        mm_env.run(days_to_expiry / 365.0, price)

        # --- Loop through each batch sequentially ---
        for batch in all_batches:
            # Deconstruct the current batch
            mm_agents_in_batch = batch[0]
            insti_agents_in_batch = batch[1]
            retail_agents_in_batch = batch[2]

            # --- Run the Market Makers from this specific batch ---
            for agent in mm_agents_in_batch:
                run_mm_step(agent, mm_env)
            
            # --- Periodically run Insti and Retail agents from this specific batch ---
            if (step + 1) % decision_frequency == 0:

                # Run INSTITUTIONAL AGENTS in this batch
                for agent in insti_agents_in_batch:
                    env = insti_env_map[agent.agent_id]
                    current_observation = env.get_state()
                    action = agent.choose_action(current_observation)
                    new_observation, reward, done = env.step(action)
                    agent.remember(current_observation, action, reward, new_observation, done)

                # Run RETAIL AGENTS in this batch
                for agent in retail_agents_in_batch:
                    env = retail_env_map[agent.agent_id]
                    env.make_decision()
        

        # 2. Gather all portfolio data into dictionaries
        all_insti_portfolios = {env.agent.agent_id: env.portfolio_value for env in insti_envs}
        all_mm_portfolios = {agent.agent_id: agent.broker.portfolio_value for agent in mm_agents}
        all_retail_portfolios = {env.agent.agent_id: calculate_retail_portfolio_value(env, price) for env in retail_envs}

        # 3. Call the single update function with all the new data
        update_data(
            price=price,
            insti_portfolios=all_insti_portfolios,
            mm_portfolios=all_mm_portfolios,
            retail_portfolios=all_retail_portfolios
        )
    # --- E. END-OF-DAY LEARNING & REPORTING ---
    print(f"\n--- Day {i+1} Finished | Closing Price: {daily_price_path[-1]:.2f} ---")
    
    mm_env.indicator()
    
    # Institutional agents learn from the day's experiences
    for agent in insti_agents:
        agent.learn()
    
    # Save models periodically
    # Save models every 10 days (rotating slots 0-9)
    if (i + 1) % 10 == 0:
        slot = ((i + 1) // 10) % 10  # This gives unique slots: 1, 2, 3, ..., 9, 0
        print(f"--- Saving DDPG Models at Day {i+1} (Slot {slot}) ---")
        for agent in insti_agents:
            agent.save_models(slot)  
    
    # ... rest of your reporting code ...
    print("  --- Institutional Agent Portfolios ---")
    for env in insti_envs:
        print(f"    {env.agent.agent_id}: Value: {env.portfolio_value:,.2f}, Cash: {env.cash_balance:,.2f}, Pos: {dict(env.portfolio)}")
    
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
        opening_price = daily_price_path[0]
        print(f"    Settlement: Final Stock Price = {final_price}")
        for env in insti_envs: env.settle_at_expiry(final_price)
        for env in retail_envs: env.settle_at_expiry(final_price)
        for agent in mm_agents: agent.action_at_expiry(opening_price, final_price)

        mm_env.action_at_expiry()
        
        days_to_expiry = option_cycle_days
        
print("\n--- Simulation Complete ---")

# Final model save
print("--- Saving Final Models ---")
for agent in insti_agents:
    agent.save_models(n_episodes)