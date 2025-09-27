import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Concatenate

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, name='critic', checkpoint_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=12, n_tickers=11, name='actor', checkpoint_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.n_tickers = n_tickers
        self.model_name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_ddpg.h5')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        
        # Head 1: For choosing the ticker (multiple choice)
        self.ticker_choice = Dense(self.n_tickers, activation='softmax')
        # Head 2: For choosing the size and direction (slider)
        self.size_and_direction = Dense(1, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        
        tickers = self.ticker_choice(x)
        size = self.size_and_direction(x)
        
        # Combine the two heads into a single output tensor
        return tf.concat([tickers, size], axis=1)