import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from .networks_sac import spreadActorNetwork, spreadCriticNetwork, spreadValueNetwork
from collections import defaultdict
class MarketMakerAgent:
    def __init__(self, id):
        self.agent_id = id
        self.env = None # This will be connected by main.py
        self.value = spreadValueNetwork()
        self.target_value = spreadValueNetwork()
        self.critic_1 = spreadCriticNetwork()
        self.critic_2 = spreadCriticNetwork()
        self.actor = spreadActorNetwork()
        self.value.compile(optimizer=Adam(learning_rate=0.001))
        self.target_value.compile(optimizer=Adam(learning_rate=0.001))
        self.critic_1.compile(optimizer=Adam(learning_rate=0.001))
        self.critic_2.compile(optimizer=Adam(learning_rate=0.001))
        self.actor.compile(optimizer=Adam(learning_rate=0.001))
        self.capital = 10000000
        self.inventory = defaultdict(int)
        self.tau = 0.005
        self.batch_size = 500
        self.memory_size = 2000
        self.states = [0] * self.memory_size
        self.actions = [0] * self.memory_size
        self.log_probs = [0] * self.memory_size
        self.rewards = [0] * self.memory_size
        self.new_states = [0] * self.memory_size
        self.t = 0

    def collect(self):
        # This is your partner's original logic
        i = self.t % self.memory_size
        ob = self.env.get_orderbook()
        highest_bid, lowest_ask = self.env.get_highestbid_lowestask(ob, self.agent_id)
        current_inventory_for_this_ticker = self.inventory.get(ob.ticker_id, 0)
        state = [highest_bid, lowest_ask, lowest_ask - highest_bid,(lowest_ask + highest_bid)/2, self.capital, current_inventory_for_this_ticker]
    
        self.states[i] = state
        action, logprob = self.get_action(state)
        self.actions[i] = action
        self.log_probs[i] = logprob
        bid_price, ask_price, bid_size, ask_size = self.decide_prices_sizes(action, highest_bid, lowest_ask)
        executed_bid_price, executed_ask_price, executed_bid_size, executed_ask_size = self.env.update_book(ob, bid_price, ask_price, bid_size, ask_size, self.agent_id)
        PL, size_diff, reward = self.env.get_reward(executed_ask_price, executed_ask_size, executed_bid_price, executed_bid_size, highest_bid, lowest_ask)
        self.rewards[i] = reward
        self.capital += PL
        self.inventory[ob.ticker_id] += size_diff
        new_highest_bid, new_lowest_ask = self.env.get_highestbid_lowestask(ob, self.agent_id)
        new_inventory = self.inventory.get(ob.ticker_id, 0)
        new_state = [new_highest_bid, new_lowest_ask, new_lowest_ask - new_highest_bid, 
                 (new_lowest_ask + new_highest_bid)/2, self.capital, new_inventory]
        self.new_states[i] = new_state
        self.t += 1

    def get_action(self, state):
        tfstate = tf.convert_to_tensor([state], dtype=tf.float32)
        action, logprob = self.actor.sample_normal(tfstate, reparameterize=False)
        return action.numpy(), logprob.numpy()
    
    def decide_prices_sizes(self, action, highest_bid, lowest_ask):
        action_values = action[0]
        bid_price = highest_bid + action_values[0] * (lowest_ask - highest_bid) / 2
        ask_price = lowest_ask - action_values[1] * (lowest_ask - highest_bid) / 2
        bid_size = int(action_values[2] * 100)
        ask_size = int(action_values[3] * 100)
        return bid_price, ask_price, bid_size, ask_size
    
    def sample_batch(self):
        states, actions, rewards, new_states = [], [], [], []
        max_mem = min(self.t, self.memory_size)
        for _ in range(self.batch_size):
            j = np.random.randint(0, max_mem)
            states.append(self.states[j])
            actions.append(self.actions[j])
            rewards.append(self.rewards[j])
            new_states.append(self.new_states[j])
        return states, actions, rewards, new_states
    
    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_value.set_weights(weights)

    def learn(self):
        self.scale = 2.0
        self.gamma = 0.99
        self.alpha = 0.05
        self.batch_size = 500

        #states, actions, log_probs, rewards, new_states = self.learn()
        for k in range(0,10):
            # if k%4==0:
            #     print(k, end=" ")
            states, actions,  rewards, new_states = self.sample_batch()
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            actions = tf.squeeze(actions,1)
            #log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)
            #print(states.shape, actions.shape, rewards.shape, states_.shape)
            #print("conversions done")

            #print("optimizers set")

            with tf.GradientTape() as tape:
                value = tf.squeeze(self.value(states), 1)    #value for state
                value_ = tf.squeeze(self.target_value(states_), 1)    #target value for state
                #print("got values")
                #print(value.shape, value_.shape)

                current_policy_actions, log_probs = self.actor.sample_normal(states,    # get the actions and their probabilites of getting selected
                                                            reparameterize=False)
                #print(current_policy_actions.shape, log_probs.shape)
                #current_policy_actions = tf.squeeze(current_policy_actions, 1)
                #log_probs = tf.squeeze(log_probs, 1)
                #print("squeezed acts, logs")
                #print(current_policy_actions.shape, log_probs.shape)
                q1_new_policy = self.critic_1(states, current_policy_actions)     #critic 1 for state action
                q2_new_policy = self.critic_2(states, current_policy_actions)    #critic 2 for state action
                #print(q1_new_policy.shape, q2_new_policy.shape)
                #print("got critics")
                critic_value = tf.squeeze(
                                    tf.math.minimum(q1_new_policy, q2_new_policy), 1)     #get min of critics

                value_target = critic_value - self.alpha*log_probs           #tells difference in how good action is and its probability of getting chosen
                                                                #i.e. how far critic and actor are
                value_loss = 0.5 * tf.reduce_mean(tf.square(value - value_target))


            value_network_gradient = tape.gradient(value_loss,
                                                    self.value.trainable_variables)

            # for i, g in enumerate(value_network_gradient):
            #     print(f"Grad {i}: {g}")

            self.value.optimizer.apply_gradients(zip(
                          value_network_gradient, self.value.trainable_variables))
            #print("value update")


            with tf.GradientTape() as tape:
                # in the original paper, they reparameterize here. We don't implement
                # this so it's just the usual action.
                new_policy_actions, log_probs = self.actor.sample_normal(states,    #again action and and their logprobs
                                                    reparameterize=True)
                log_probs = tf.squeeze(log_probs, 1)
                q1_new_policy = self.critic_1(states, new_policy_actions)    #critic 1
                q2_new_policy = self.critic_2(states, new_policy_actions)    #critic 2
                critic_value = tf.squeeze(tf.math.minimum(                   #their min
                                            q1_new_policy, q2_new_policy), 1)

                actor_loss = self.alpha*log_probs - critic_value                       # -ve of value_target which we want to maximizme and this we want to minimize
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss,
                                                self.actor.trainable_variables)
            # for i, g in enumerate(actor_network_gradient):
            #     print(f"Grad {i}: {g}")
            self.actor.optimizer.apply_gradients(zip(
                            actor_network_gradient, self.actor.trainable_variables))
            #print("actor update")

            with tf.GradientTape(persistent=True) as tape:
                done = 0
                # I didn't know that these context managers shared values?
                self.scale=0.5
                q_hat = self.scale*rewards + self.gamma*value_*(1-done)
                q1_old_policy = tf.squeeze(self.critic_1(states, actions), 1)
                q2_old_policy = tf.squeeze(self.critic_2(states, actions), 1)
                critic_1_loss = 0.5 * tf.reduce_mean(tf.square(q1_old_policy - q_hat))
                critic_2_loss = 0.5 * tf.reduce_mean(tf.square(q2_old_policy - q_hat))

            critic_1_network_gradient = tape.gradient(critic_1_loss,
                                            self.critic_1.trainable_variables)
            critic_2_network_gradient = tape.gradient(critic_2_loss,
                self.critic_2.trainable_variables)

            # for i, g in enumerate(critic_1_network_gradient):
            #     print(f"Grad {i}: {g}")
            # for i, g in enumerate(critic_2_network_gradient):
            #     print(f"Grad {i}: {g}")

            self.critic_1.optimizer.apply_gradients(zip(
                critic_1_network_gradient, self.critic_1.trainable_variables))
            self.critic_2.optimizer.apply_gradients(zip(
                critic_2_network_gradient, self.critic_2.trainable_variables))
            #print("critics update")

            self.update_network_parameters()
            #print("target value update")
        #print("\n")

    # def get_action(self, hb, la):
    #     self.state = [ hb, la, la - hb, (la + hb)/2, self.capital, self.size]
    #     state = tf.convert_to_tensor(self.state)
    #     state = tf.expand_dims(state, axis = 0)
    #     mu, sigma = self.actor(state)

    #     bid_price = hb + mu[0][0] * (la - hb) / 2
    #     ask_price = la - mu[0][1] * (la - hb) / 2
    #     #print("price done")
    #     bid_size = tf.math.floor(mu[0][2] * 100)
    #     ask_size = tf.math.floor(mu[0][3] * 100)

    #     return bid_price, ask_price, bid_size, ask_size
    def _calculate_portfolio_value(self, central_market):
        
        value = self.capital
        for ticker_name,quantity in self.inventory.items():
            if quantity == 0:
                continue
            book = central_market.get_book(ticker_name)
            if book:
                price  = 0
                if quantity>0:
                    bids=book.get_bids('MarketMaker')
                    if bids:
                        price = bids[0][0]
                    if price > 0:
                        value += quantity * price 
        return value