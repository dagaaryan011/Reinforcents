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
MODEL_BASE_PATH = r"D:\NetworkPrediction\data\models\retail"
PERSONALITY_MODEL_NAMES = {
    'Momentum Trader': ['macd', 'rsi', 'stoch'],
    'Trend Follower':  ['rsi', 'adx', 'status'],
    'Hybrid Trader':   ['macd', 'stoch', 'status'],
    'All-Rounder':     ['macd', 'rsi', 'stoch', 'adx', 'status']
}

# The total number of features in the input vector.
NUM_FEATURES = len(FEATURE_LIST)


# --- 2. File & Directory Paths ---

MODEL_PATH = "data/models/"
TRAINING_DATA_PATH = "data/training_data/"
PRICE_DATA_CSV = r"D:\NetworkPrediction\misceleaneous\test_price_data.csv"
BASE_path_INSTI_AGENT = r"D:\NetworkPrediction\data\models\Insti" 
MODEL_DIR_MM  = r"D:\NetworkPrediction\data\models\MarketMaker"
Trades_CSV_path =r"D:\NetworkPrediction\src\market\master_trades.csv"
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



#insti

LONG_TERM_INDICATORS = ['rsi', 'stoch', 'adx', 'status']
SHORT_TERM_INDICATORS = ['macd', 'rsi', 'stoch']

NUM_LONG_TERM_FEATURES = len(LONG_TERM_INDICATORS)
NUM_SHORT_TERM_FEATURES = len(SHORT_TERM_INDICATORS)
NUM_LONG_TERM_OUTPUTS = 3 # Trend classes
NUM_ACTIONS = 11 # Example: Number of tickers the DDPG agent can trade