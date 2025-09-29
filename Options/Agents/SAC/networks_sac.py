import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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
        self.fc2 = Dense(16, activation='relu')
        self.mu = Dense(self.n_actions, activation='relu')
        self.sigma = Dense(self.n_actions, activation='relu')

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
        action = probabilities.sample()
        # action = tf.clip_by_value(action, 0.0, 1.0)
        log_probs = probabilities.log_prob(action)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action, log_probs
    

actor = spreadActorNetwork()
critic_1 = spreadCriticNetwork()
critic_2 = spreadCriticNetwork()
value = spreadValueNetwork()
target_value = spreadValueNetwork()

target_value_optimizer = Adam(learning_rate=0.001)
value_optimizer = Adam(learning_rate=0.001)
critic_1_optimizer = Adam(learning_rate=0.001)
critic_2_optimizer = Adam(learning_rate=0.001)
actor_optimizer = Adam(learning_rate=0.001)



