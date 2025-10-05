# In pipeline.py

import tensorflow as tf
import numpy as np
import os
from collections import deque


from functions import (
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


from model import SEQUENCE_LENGTH, NUM_FEATURES

class Pipeline:
    def __init__(self):
        model_path = "models/model_rnn_1.keras"
        print("--- Loading model for inference ---")
        self.main_network = tf.keras.models.load_model(model_path)

        self.trend = []
        self.macd_history = []
        self.rsi_history = []
        self.k_history = []
        self.d_history = []
        self.DMpos_history = []
        self.DMneg_history = []
        self.true_range_history = []
        self.dx_history = []
        
        self.feature_window = [] 

    def preprocess_input(self, price: float):
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

        
        macd_sig = macd_signal(self.macd_history)
        rsi_sig = get_rsi_conviction(self.rsi_history)
        stoch_sig = get_stoch_conviction(self.k_history, self.d_history)
        status_sig = get_current_status(self.trend)
        adx_sig = get_ADX(self.dx_history)
        
        current_features = [
            macd_sig if macd_sig is not None else 0.0,
            rsi_sig if rsi_sig is not None else 0.0,
            stoch_sig if stoch_sig is not None else 0.0,
            status_sig if status_sig is not None else 0.0,
            adx_sig if adx_sig is not None else 0.0
        ]
        
        self.feature_window.append(current_features)

    def inference(self):
        if len(self.feature_window) < SEQUENCE_LENGTH:
            return None

        input_sequence_list = self.feature_window[-SEQUENCE_LENGTH:]
        input_sequence = np.array(input_sequence_list).reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
        prediction_probabilities = self.main_network.predict(input_sequence, verbose=0)
        
        return prediction_probabilities

    
    def predict(probabilities):
        
        if probabilities is None:
            return None
        
        predicted_class = np.argmax(probabilities)
        signal = predicted_class - 1
        return signal

    def train(self, matrix):
        
        train_x = np.array([row[0] for row in matrix])
        train_y = np.array([row[1] + 1 for row in matrix])
        self.main_network.fit(
            train_x,
            train_y,
            epochs=10,
            validation_split=0.2
        )
        self.main_network.save("models/model_rnn_1.keras")