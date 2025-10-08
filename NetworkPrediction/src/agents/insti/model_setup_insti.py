
import torch
import torch.nn as nn
from config import (
    SEQUENCE_LENGTH, 
    NUM_LONG_TERM_FEATURES, 
    NUM_SHORT_TERM_FEATURES, 
    NUM_LONG_TERM_OUTPUTS, 
    NUM_ACTIONS
)

class LongTermModel(nn.Module):
    """ The 'Navigator' network for identifying the overall trend. """
    def __init__(self):
        super(LongTermModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=NUM_LONG_TERM_FEATURES, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc_out = nn.Linear(32, NUM_LONG_TERM_OUTPUTS)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        x = self.dropout1(lstm_out)
        lstm_out, _ = self.lstm2(x)
        last_time_step_out = lstm_out[:, -1, :]
        logits = self.fc_out(last_time_step_out)
        return logits

class ShortTermModel(nn.Module):
    """ The 'Helmsman' network for making the final trade decision. """
    def __init__(self):
        super(ShortTermModel, self).__init__()
        combined_input_size = NUM_SHORT_TERM_FEATURES + NUM_LONG_TERM_OUTPUTS
        
        self.lstm1 = nn.LSTM(input_size=combined_input_size, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, NUM_ACTIONS)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        x = self.dropout1(lstm_out)
        lstm_out, _ = self.lstm2(x)
        last_time_step_out = lstm_out[:, -1, :]
        x = self.fc1(last_time_step_out)
        x = self.relu(x)
        logits = self.fc_out(x)
        return logits