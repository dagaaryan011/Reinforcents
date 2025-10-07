# In src/agent/agent.py

import numpy as np
from collections import deque
import tensorflow as tf
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.model_resources.model import Network_Utils

# Import utilities and configuration
from ...tools.functions import (
    Indicators, 
    macd_signal, 
    get_rsi_conviction, 
    get_stoch_conviction, 
    get_current_status, 
    get_DMs, 
    get_true_range, 
    get_DX, 
    get_ADX
)
from config import SEQUENCE_LENGTH,FEATURE_LIST,AGENT_FPERSONALITIES

class Agent:
    def __init__(self, agent_id: int, model_path:str):
        
        self.agent_id = agent_id
        self.main_network = Network_Utils()
        self.main_network.load_model(model_path)
        self.indicator_focus = random.choice(AGENT_FPERSONALITIES)
        
        
        rand_rsi = np.random.rand(3); self.rsi_coeffs = rand_rsi / np.sum(rand_rsi)
        rand_stoch = np.random.rand(3); self.stoch_coeffs = rand_stoch / np.sum(rand_stoch)
            
        print(f"--- Initialized Agent {self.agent_id} with focus: {self.indicator_focus} ---")

        # --- Initialize State Histories ---
        self.trend = []
        self.macd_history = []
        self.rsi_history = []
        self.k_history = []
        self.d_history = []
        self.DMpos_history = []
        self.DMneg_history = []
        self.true_range_history = []
        self.dx_history = []
        
        
        
        self.feature_window = deque(maxlen=SEQUENCE_LENGTH)

    def preprocess_input(self, price: float):
        """
        Takes a new price, updates histories, and generates the latest feature
        vector, masked according to the agent's focus.
        """
        
        self.trend.append(price)
        self.macd_history.append(Indicators.MACD(self.trend))
        self.rsi_history.append(Indicators.Smooth_RSI(self.trend))
        self.k_history.append(Indicators.Stoch_Oscilator(self.trend))
        DMpos, DMneg = get_DMs(self.trend)
        self.DMneg_history.append(DMneg)
        self.DMpos_history.append(DMpos)
        
        if len(self.k_history) >= 3:
            last_three_k = self.k_history[-3:]
            if all(k is not None for k in last_three_k):
                d_value = sum(last_three_k) / 3
                self.d_history.append(d_value)
            else:
                self.d_history.append(None)
        else:
             self.d_history.append(None)
        self.true_range_history.append(get_true_range(self.trend))      
        self.dx_history.append(get_DX(self.DMpos_history, self.DMneg_history, self.true_range_history))

        
        full_feature_set = {
            'macd': macd_signal(self.macd_history),
            'rsi': get_rsi_conviction(self.rsi_history, coeffs=self.rsi_coeffs),
            'stoch': get_stoch_conviction(self.k_history, self.d_history, coeffs=self.stoch_coeffs),
            'status': get_current_status(self.trend),
            'adx': get_ADX(self.dx_history)
        }
        
        
        final_features = []
        for indicator in FEATURE_LIST:
            if indicator in self.indicator_focus:
                
                value = full_feature_set[indicator]
                final_features.append(value if value is not None else 0.0)
            else:
                
                final_features.append(0.0)
        
        self.feature_window.append(final_features)

    # In your Agent class (e.g., src/agents/retail/agent_retail.py)

    def inference(self):
        """ Runs the model on the current feature sequence. """
        if len(self.feature_window) < SEQUENCE_LENGTH:
            return None

        # 1. Create the NumPy array as before
        input_sequence_np = np.array(list(self.feature_window), dtype=np.float32)

        # 2. Add the batch dimension (shape becomes (1, 30, 5))
        input_sequence_np = np.expand_dims(input_sequence_np, axis=0)

        # --- 3. THIS IS THE FIX: Convert NumPy array to a PyTorch Tensor ---
        input_tensor = torch.from_numpy(input_sequence_np)

        # 4. Pass the TENSOR to your network's output method
        return self.main_network.output(input_tensor)

    # @staticmethod
    def predict(probabilities: torch.Tensor):
        """ Converts raw probabilities (as a PyTorch Tensor) into a signal. """
        if probabilities is None:
            return None

        predicted_class = torch.argmax(probabilities)

        
        signal = predicted_class.item() - 1

        return signal
