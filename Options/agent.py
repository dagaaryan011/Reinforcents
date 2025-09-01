import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer # Renamed for clarity
from network import ActorNetwork, CriticNetwork

import os

class Agent:
    def __init__(self, input_dims, env, n_actions=12, n_tickers=11,
                 alpha=0.001, beta=0.002, gamma=0.99, tau=0.005,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=64, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_tickers = n_tickers
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(n_actions=n_actions, n_tickers=n_tickers, name='actor', fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.critic = CriticNetwork(name='critic', fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_actor = ActorNetwork(n_actions=n_actions, n_tickers=n_tickers, name='target_actor', fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.target_critic = CriticNetwork(name='target_critic', fc1_dims=fc1_dims, fc2_dims=fc2_dims)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        
        if not evaluate:
            # Add noise only to the continuous part (the last element)
            noise_tensor = tf.random.normal(shape=(1,), mean=0.0, stddev=self.noise)
            # Create a zero-padding for the discrete part
            zero_padding = tf.zeros(shape=(self.n_tickers,))
            # Combine to make a noise vector of the correct shape
            full_noise = tf.concat([zero_padding, noise_tensor], axis=0)
            actions += full_noise

        # Clip only the continuous part
        ticker_probs = actions[0,:self.n_tickers]
        size_action = tf.clip_by_value(actions[0,self.n_tickers:], self.min_action, self.max_action)

        # Recombine and return the final action tensor
        final_action = tf.concat([ticker_probs, size_action], axis=0)
        return final_action

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            critic_value_ = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * critic_value_ * (1 - done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = -self.critic(states, new_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)