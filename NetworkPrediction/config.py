import os

# --- 0. Kaggle Base Path ---
# All other paths will be constructed from this base to ensure portability.
KAGGLE_PROJECT_ROOT = "/kaggle/working/Reinforcents/NetworkPrediction"


# --- 1. Model & Data Configuration ---

# The number of time steps the RNN looks at for each prediction.
SEQUENCE_LENGTH = 30

# The master list of all possible indicators in the correct order.
# This is crucial for the agent's feature masking.
FEATURE_LIST = ['macd', 'rsi', 'stoch', 'status', 'adx']

# Agent personalities defined by the features they use.
AGENT_PERSONALITIES = [
    ['macd', 'rsi', 'stoch'],              # Momentum Trader
    ['rsi', 'adx', 'status'],               # Trend Follower
    ['macd', 'stoch', 'status'],            # Hybrid Trader
    ['macd', 'rsi', 'stoch', 'adx', 'status'] # All-Rounder
]

# The total number of features in the input vector.
NUM_FEATURES = len(FEATURE_LIST)


# --- 2. File & Directory Paths ---

# Base path for retail agent models
MODEL_BASE_PATH_RETAIL = os.path.join(KAGGLE_PROJECT_ROOT, "data/models/retail")

# A dictionary that maps agent personality names to their full .pth model file paths
PERSONALITY_MODEL_PATHS = {
    # The key (e.g., 'Momentum Trader') is used to generate the filename.
    # Resulting path: /kaggle/working/Reinforcents/NetworkPrediction/data/models/retail/Momentum Trader.pth
    name: os.path.join(MODEL_BASE_PATH_RETAIL, f"{name.replace(' ', '_')}.pth") 
    for name in ['Momentum Trader', 'Trend Follower', 'Hybrid Trader', 'All-Rounder']
}
# Note: I've replaced spaces with underscores (e.g., 'Momentum_Trader.pth') 
# as it's a safer practice for filenames.

# General model directory
MODEL_PATH = os.path.join(KAGGLE_PROJECT_ROOT, "data/models/")

# Training data path
TRAINING_DATA_PATH = os.path.join(KAGGLE_PROJECT_ROOT, "data/training_data")

# Price data CSV file
PRICE_DATA_CSV = os.path.join(KAGGLE_PROJECT_ROOT, "misceleaneous/test_price_data.csv")

# Path for institutional agent models
BASE_path_INSTI_AGENT = os.path.join(KAGGLE_PROJECT_ROOT, "data/models/insti")

# Path for Market Maker models
MODEL_DIR_MM = os.path.join(KAGGLE_PROJECT_ROOT, "data/models/MarketMaker")

# Path for the master trades CSV log
TRADES_CSV_PATH = os.path.join(KAGGLE_PROJECT_ROOT, "src/market/master_trades.csv")


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

# Lot size for the options
LOT_SIZE = 50


# --- 5. Institutional Agent Specifics ---

LONG_TERM_INDICATORS = ['rsi', 'stoch', 'adx', 'status']
SHORT_TERM_INDICATORS = ['macd', 'rsi', 'stoch']

NUM_LONG_TERM_FEATURES = len(LONG_TERM_INDICATORS)
NUM_SHORT_TERM_FEATURES = len(SHORT_TERM_INDICATORS)
NUM_LONG_TERM_OUTPUTS = 3 # Trend classes
NUM_ACTIONS = 11 # Example: Number of tickers the DDPG agent can trade