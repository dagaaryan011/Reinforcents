# In src/agent/agent.py

import numpy as np
from collections import deque
import tensorflow as tf
import random

# Import utilities and configuration
from ..tools.functions import (
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
    def __init__(self, agent_id: int, model):
        
        self.agent_id = agent_id
        self.main_network = model
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

    def inference(self):
        """ Runs the model on the current feature sequence. """
        if len(self.feature_window) < SEQUENCE_LENGTH:
            return None

        num_features = len(FEATURE_LIST)
        input_sequence = np.array(list(self.feature_window), dtype=np.float32).reshape(1, SEQUENCE_LENGTH, num_features)
        return self.main_network.predict(input_sequence, verbose=0)

    
    def predict(probabilities):
        """ Converts raw probabilities into the final trading signal (-1, 0, 1). """
        if probabilities is None:
            return None
        return np.argmax(probabilities) - 1