# In main.py

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

# Import our custom classes from the 'src' directory
from src.market.exchange import MarketExchange
from src.agents.agent_retail import Agent
from src.agents.agent_retail_env import AgentEnvironment
import sys
print(sys.argv)

# Import constants from our central config file
from config import (
    MODEL_PATH, 
    NUM_AGENTS_IN_SIMULATION, 
    PRICE_DATA_CSV, 
    DECISION_FREQUENCY
)

if __name__ == "__main__":
    # --- 1. SETUP THE SIMULATION ENVIRONMENT ---
    print("--- Setting up simulation environment... ---")
    
    
    exchange = MarketExchange() 
    
    
    agent_environments = {}
    
    for i in range(NUM_AGENTS_IN_SIMULATION):
        print(f"Loading model and creating agent {i}...")
        
        # Load the pre-trained model for this specific agent
        model_file = r"D:\NetworkPrediction\data\models\model_rnn_1.keras"
        model = tf.keras.models.load_model(model_file)
        
        # Create the agent's "brain"
        agent = Agent(agent_id=i, model=model)
        
        # Create the agent's "body" and link it to the brain and the exchange
        env = AgentEnvironment(agent=agent, exchange=exchange)
        agent_environments[agent.agent_id] = env

    # Load the historical price data to drive the simulation
    
        price_data = pd.read_csv(PRICE_DATA_CSV)['price'].tolist()
    
        

    # --- 2. RUN THE SIMULATION LOOP ---
    print("\n--- Starting simulation... ---")
    for step, price in enumerate(tqdm(price_data, desc="Running Simulation")):
        
        # A. Update the market with the new underlying price
        exchange.update_market(price)
        
        # B. Every N steps, allow agents to make a decision
        if (step + 1) % DECISION_FREQUENCY == 0:
            print(f"\n--- Decision Point: Step {step + 1} | Price: {price:.2f} ---")
            for env in agent_environments.values():
                env.step() # Each environment runs its agent's full decision cycle
    
    # --- 3. SIMULATION WRAP-UP ---
    print("\n--- Simulation Complete ---")
    # (Here you would add logic to settle final positions and plot results)
    for i, env in enumerate(agent_environments):
        print(f"Agent {i} Final Cash Balance: {env.cash_balance:.2f}")