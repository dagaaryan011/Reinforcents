import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense

class spreadCriticNetwork(keras.Model):
    def __init__(self):
        super(spreadCriticNetwork, self).__init__()
        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.q = Dense(1, activation=None)
    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        return self.q(x)

class spreadValueNetwork(keras.Model):
    def __init__(self):
        super(spreadValueNetwork, self).__init__()
        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.v = Dense(1, activation=None)
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.v(x)

class spreadActorNetwork(keras.Model):
    def __init__(self):
        super(spreadActorNetwork, self).__init__()
        self.n_actions = 4
        self.noise = 1e-6
        self.fc1 = Dense(32, activation='relu')
        self.fc2 = Dense(32, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = tf.clip_by_value(sigma, self.noise, 1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        actions = probabilities.sample()
        action = tf.math.tanh(actions)
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action, log_probs
    
class bookCriticNetwork(keras.Model):
    def __init__(self):
        super(spreadCriticNetwork, self).__init__()
        self.fc1 = Dense(16, activation='relu')
        self.fc2 = Dense(8, activation='relu')
        self.fc3 = Dense(4, activation='relu')
        self.fc4 = Dense(2, activation='relu')
        self.c = Dense(1, activation='softmax')
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.c(x)
    
class bookActorNetwork(keras.Model):
    def __init__(self):
        super(spreadActorNetwork, self).__init__()
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(128, activation='relu')
        self.fc4 = Dense(64, activation='relu')
        self.fc5 = Dense(32, activation='relu')
        self.fc6 = Dense(16, activation='relu')
        self.a = Dense(20, activation='softmax')
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return self.a(x)