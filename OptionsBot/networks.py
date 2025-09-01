import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense

class spreadCriticNetwork(keras.Model):
    def __init__(self): #, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(spreadCriticNetwork, self).__init__()
        self.layer1_dims = 32
        self.layer2_dims = 32
        self.n_actions = 4
        # self.model_name = name
        # self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.layer1_dims, activation='relu')
        self.fc2 = Dense(self.layer2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.fc1(tf.concat([state, action], axis=1))
        action_value = self.fc2(action_value)

        q = self.q(action_value)

        return q

class spreadValueNetwork(keras.Model):
    def __init__(self): #, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
        super(spreadValueNetwork, self).__init__()
        self.layer1_dims = 32
        self.layer2_dims = 32
        # self.model_name = name
        # self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = Dense(self.layer1_dims, activation='relu')
        self.fc2 = Dense(self.layer2_dims, activation='relu')
        self.v = Dense(1, activation=None)

    def call(self, state):
        state_value = self.fc1(state)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v

class spreadActorNetwork(keras.Model):
    def __init__(self):
        super(spreadActorNetwork, self).__init__()
        self.layer1_dims = 32
        self.layer2_dims = 32
        self.n_actions = 4
        #self.model_name = name
        #self.checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.noise = 1e-1

        self.fc1 = Dense(self.layer1_dims, activation='relu')
        self.fc2 = Dense(self.layer2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='sigmoid')
        self.sigma = Dense(self.n_actions, activation='sigmoid')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)


        if reparameterize:
            actions = probabilities.sample() # + something else if you want to implement
        else:
            actions = probabilities.sample()

        

        actions = tf.sigmoid(actions)

        log_probs = probabilities.log_prob(actions)

        epsilon = 1e-6
        squash_correction = tf.math.log(tf.clip_by_value(actions * (1 - actions), epsilon, 1.0))

        log_probs = log_probs - squash_correction

        log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)

        # actions = tf.sigmoid(actions)

        # log_probs = probabilities.log_prob(actions)

        # # # Jacobian of the sigmoid function
        # log_probs -= tf.math.log(actions * (1 - actions) + self.noise)  # Add small noise for numerical stability

        # log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)  # Sum log-probabilities over all actions


        # dist = tfp.distributions.Normal(mu, sigma)

        # # Sample the action (either with or without reparameterization)
        # action = dist.sample()  # Always sample actions directly

        # # If using reparameterization, sample with the trick
        # if reparameterize:
        #     action = action  # Reparameterized, though same as normal sample in this case
        # else:
        #     action = action  # Non-reparameterized, could be custom if you change logic

        # # Squash action with sigmoid (to map action to [0, 1])
        # action = tf.sigmoid(action)

        # # Log-probability calculation with the tanh adjustment (if necessary)
        # log_probs = dist.log_prob(action)

        # # Jacobian of the sigmoid function
        # log_probs -= tf.math.log(action * (1 - action) + self.noise)  # Add small noise for numerical stability

        # log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)  # Sum log-probabilities over all actions


        # action = tf.math.tanh(actions)*1
        # log_probs = probabilities.log_prob(actions)
        # log_probs -= tf.math.log(1-tf.math.pow(action,2)+self.noise)
        # log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        # return action, log_probs
        return actions, log_probs

