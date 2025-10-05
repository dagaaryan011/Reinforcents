# In config.py

# --- 1. Model & Data Configuration ---

# The number of time steps the RNN looks at for each prediction.
SEQUENCE_LENGTH = 30

# The master list of all possible indicators in the correct order.
# This is crucial for the agent's feature masking.
FEATURE_LIST = ['macd', 'rsi', 'stoch', 'status', 'adx']
#agent personalities
AGENT_FPERSONALITIES = [
    ['macd', 'rsi', 'stoch'],              # Momentum Trader
    ['rsi', 'adx', 'status'],               # Trend Follower
    ['macd', 'stoch', 'status'],            # Hybrid Trader
    ['macd', 'rsi', 'stoch', 'adx', 'status'] # All-Rounder
]
# The total number of features in the input vector.
NUM_FEATURES = len(FEATURE_LIST)


# --- 2. File & Directory Paths ---

MODEL_PATH = "data/models/"
TRAINING_DATA_PATH = "data/training_data/"
PRICE_DATA_CSV = r"D:\NetworkPrediction\misceleaneous\test_price_data.csv"


# --- 3. Training Hyperparameters ---

EPOCHS = 20
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2


# --- 4. Simulation Parameters ---

# The number of unique agent models to train.
NUM_AGENTS_TO_TRAIN = 10

# The number of agents to run in the main simulation.
NUM_AGENTS_IN_SIMULATION = 10

# Agents will make a prediction every N steps.
DECISION_FREQUENCY = 30

#lot size for the options
LOT_SIZE = 50