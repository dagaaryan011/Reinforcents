import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from .networks_sac import actor, critic_1, critic_2, value, target_value, actor_optimizer, critic_1_optimizer, critic_2_optimizer, value_optimizer, target_value_optimizer
from collections import defaultdict


checkpoint_actor = "C:\ProjectX\Options\Agents\SAC\actor.weights.h5"
checkpoint_critic_1 = "C:\ProjectX\Options\Agents\SAC\critic_1.weights.h5"
checkpoint_critic_2 = "C:\ProjectX\Options\Agents\SAC\critic_2.weights.h5"
checkpoint_value = "C:\ProjectX\Options\Agents\SAC\value.weights.h5"
checkpoint_target_value = "C:\ProjectX\Options\Agents\SAC\target_value.weights.h5"

class MarketMakerAgent:
    def __init__(self, id):
        self.agent_id = id
        self.env = None # This will be connected by main.py
        self.ticker = None
        self.value = value
        self.target_value = target_value
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.value_optimizer = value_optimizer
        self.target_value_optimizer = target_value_optimizer
        self.tau = 0.005
        self.scale = 2.0
        self.gamma = 0.99
        self.alpha = 0.05
        self.batch_size = 50
        self.memory_size = 100
        self.batch_times = 10
        self.states = [0] * self.memory_size
        self.actions = [0] * self.memory_size
        self.log_probs = [0] * self.memory_size
        self.dones = [0] * self.memory_size
        self.rewards = [0] * self.memory_size
        self.new_states = [0] * self.memory_size
        self.t = 0
        self.order_num = 0
        self.orders = [0] * self.memory_size
        self.all_orders = []
        self.load_weights()

    def load_weights(self):
        dummy_state = tf.random.normal((1, 14))
        dummy_action = tf.random.normal((1, 4))

        _ = self.actor(dummy_state)
        _ = self.critic_1(dummy_state, dummy_action)
        _ = self.critic_2(dummy_state, dummy_action)
        _ = self.value(dummy_state)
        _ = self.target_value(dummy_state)

        self.actor.load_weights('actor.weights.h5')
        self.critic_1.load_weights('critic_1.weights.h5')
        self.critic_2.load_weights('critic_2.weights.h5')
        self.value.load_weights('value.weights.h5')
        self.target_value.load_weights('target_value.weights.h5')

    def save_weights(self):
        self.actor.save_weights('actor.weights.h5')
        self.critic_1.save_weights('critic_1.weights.h5')
        self.critic_2.save_weights('critic_2.weights.h5')
        self.value.save_weights('value.weights.h5')
        self.target_value.save_weights('target_value.weights.h5')

    def set_agent_ticker(self, t):
        self.ticker = t

    def collect(self):
        
        i = self.t % self.memory_size

        state = self.env.get_current_state(self.ticker)
        self.states[i] = state
        #print(state)
        if i>0 :
            self.new_states[i-1] = state
        else :
            self.new_states[-1] = state
        action, log_prob = self.get_action(state)
        self.actions[i] = action
        self.log_probs[i] = log_prob

        bid_price, ask_price, bid_size, ask_size = self.decide_prices_sizes(action, state[2], state[3])

        buy_id = f"{self.agent_id}_BUY_{self.order_num}"
        sell_id = f"{self.agent_id}_SELL_{self.order_num}"
        executed_bid_price, executed_ask_price, executed_bid_size, executed_ask_size = self.env.update_book(self.ticker, bid_price, ask_price, bid_size, ask_size, self.agent_id, buy_id, sell_id)
        self.order_num += 1
        self.orders.append(buy_id)
        self.orders.append(sell_id)
        self.all_orders.append(buy_id)
        self.all_orders.append(sell_id)

        reward = self.env.get_reward(self.ticker, ask_price, ask_size, bid_price, bid_size, state[2], state[3])
        self.rewards[i] = reward

        self.t += 1

    def get_action(self, state):
        tfstate = tf.convert_to_tensor([state], dtype=tf.float32)
        action, logprob = self.actor.sample_normal(tfstate, reparameterize=False)
        return action.numpy(), logprob.numpy()
    
    def decide_prices_sizes(self, action, highest_bid, lowest_ask):
        action_values = action[0]
        size = min(self.env.temp_inventory, 10)
        price = min((lowest_ask - highest_bid)/2, self.env.capital)
        bid_price = highest_bid + action_values[0] * price
        ask_price = lowest_ask - action_values[1] * (lowest_ask - highest_bid) / 2
        bid_size = int(action_values[2] * 10)
        ask_size = int(action_values[3] * size)
        return bid_price, ask_price, bid_size, ask_size
    
    def sample_batch(self):
        states, actions, log_probs, rewards, new_states = [], [], [], [], []
        max_mem = min(self.t, self.memory_size)
        for _ in range(self.batch_size):
            j = np.random.randint(0, max_mem)
            states.append(self.states[j])
            actions.append(self.actions[j])
            log_probs.append(self.log_probs[j])
            rewards.append(self.rewards[j])
            new_states.append(self.new_states[j])

        return states, actions, log_probs, rewards, new_states

    def notifs(self):
        self.env.notifications(self.ticker, self.agent_id)

    def update_network_parameters(self):
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * self.tau + targets[i] * (1 - self.tau))
        self.target_value.set_weights(weights)

    def learn(self):

        for k in range(0,self.batch_times):
            # if k%4==0:
            #     print(k, end=" ")
            states, actions, log_probs, rewards, new_states = self.sample_batch()
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            actions = tf.squeeze(actions,1)
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

            with tf.GradientTape() as tape:
                value = tf.squeeze(self.value(states), 1)    #value for state
                value_ = tf.squeeze(self.target_value(states_), 1)    #target value for new state

                current_policy_actions, current_log_probs = self.actor.sample_normal(states,    # get the actions and their probabilites of getting selected
                                                            reparameterize=False)
                q1_new_policy = self.critic_1(states, current_policy_actions)     #critic 1 for state action
                q2_new_policy = self.critic_2(states, current_policy_actions)    #critic 2 for state action
                critic_value = tf.squeeze(
                                    tf.math.minimum(q1_new_policy, q2_new_policy), 1)     #get min of critics

                value_target = critic_value - self.alpha*log_probs           #tells difference in how good action is and 
                                                                                    #its probability of getting chosen
                                                                #i.e. how far critic and actor are
                value_loss = 0.5 * tf.reduce_mean(tf.square(value - value_target))

            value_network_gradient = tape.gradient(value_loss,
                                                    self.value.trainable_variables)

            self.value_optimizer.apply_gradients(zip(
                          value_network_gradient, self.value.trainable_variables))
            #print("value update")


            with tf.GradientTape() as tape:
                new_policy_actions, new_policy_log_probs = self.actor.sample_normal(states,    #again action and and their logprobs
                                                    reparameterize=True)
                new_policy_log_probs = tf.squeeze(log_probs, 1)
                q1_new_policy = self.critic_1(states, new_policy_actions)    #critic 1
                q2_new_policy = self.critic_2(states, new_policy_actions)    #critic 2
                critic_value = tf.squeeze(tf.math.minimum(                   #their min
                                            q1_new_policy, q2_new_policy), 1)

                actor_loss = self.alpha*new_policy_log_probs - critic_value   # -ve of value_target which we want to maximizme 
                                                                                #and this we want to minimize
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss,
                                                self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(
                            actor_network_gradient, self.actor.trainable_variables))

            with tf.GradientTape(persistent=True) as tape:
                self.scale=0.5
                dones = 0
                q_hat = self.scale*rewards + self.gamma*value_*(1-dones)
                q1_old_policy = tf.squeeze(self.critic_1(states, actions), 1)
                q2_old_policy = tf.squeeze(self.critic_2(states, actions), 1)
                critic_1_loss = 0.5 * tf.reduce_mean(tf.square(q1_old_policy - q_hat))
                critic_2_loss = 0.5 * tf.reduce_mean(tf.square(q2_old_policy - q_hat))

            critic_1_network_gradient = tape.gradient(critic_1_loss,
                                            self.critic_1.trainable_variables)
            critic_2_network_gradient = tape.gradient(critic_2_loss,
                self.critic_2.trainable_variables)

            self.critic_1_optimizer.apply_gradients(zip(
                critic_1_network_gradient, self.critic_1.trainable_variables))
            self.critic_2_optimizer.apply_gradients(zip(
                critic_2_network_gradient, self.critic_2.trainable_variables))
            #print("critics update")

            self.update_network_parameters()
            #print("target value update")
        #print("\n")


    