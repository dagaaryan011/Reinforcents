# config.py

# --- 1. Model & Data Configuration ---
SEQUENCE_LENGTH = 30
FEATURE_LIST = ['macd', 'rsi', 'stoch', 'status', 'adx']
AGENT_FPERSONALITIES = [
    ['macd', 'rsi', 'stoch'],
    ['rsi', 'adx', 'status'],
    ['macd', 'stoch', 'status'],
    ['macd', 'rsi', 'stoch', 'adx', 'status']
]
PERSONALITY_MODEL_NAMES = {
    'Momentum Trader': ['macd', 'rsi', 'stoch'],
    'Trend Follower':  ['rsi', 'adx', 'status'],
    'Hybrid Trader':   ['macd', 'stoch', 'status'],
    'All-Rounder':     ['macd', 'rsi', 'stoch', 'adx', 'status']
}
# --- 2. File & Directory Paths ---
# The SINGLE source of truth for the dashboard's data file path
DATA_FILE_PATH = r"D:\Reinforcents\NetworkPrediction\data\visualisation\live_data.json"

MODEL_BASE_PATH = r"D:\NetworkPrediction\data\models\retail"
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
START_CAPITAL_INSTI = 1000000
START_CAPITAL_MM = 100000
START_CAPITAL_RETAIL = 100000
NUM_AGENTS_TO_TRAIN = 10
NUM_AGENTS_IN_SIMULATION = 10
DECISION_FREQUENCY = 30
LOT_SIZE = 50

# --- 5. Insti Agent Specifics ---
LONG_TERM_INDICATORS = ['rsi', 'stoch', 'adx', 'status']
SHORT_TERM_INDICATORS = ['macd', 'rsi', 'stoch']
NUM_LONG_TERM_FEATURES = len(LONG_TERM_INDICATORS)
NUM_SHORT_TERM_FEATURES = len(SHORT_TERM_INDICATORS)
NUM_LONG_TERM_OUTPUTS = 3
NUM_ACTIONS = 11