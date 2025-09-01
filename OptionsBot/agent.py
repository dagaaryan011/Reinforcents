import os
import random
import numpy as np
import math
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
#from buffer import ReplayBuffer
from networks import spreadActorNetwork, spreadCriticNetwork, spreadValueNetwork
from exchange import MarketExchange
from orderbook import Order, Trade, Side

class Agent:
    def __init__(self):
        self.agent_id = 'maker'
        self.value = spreadValueNetwork()
        self.target_value = spreadValueNetwork()
        self.critic_1 = spreadCriticNetwork()
        self.critic_2 = spreadCriticNetwork()
        self.actor = spreadActorNetwork()
        self.exchange = MarketExchange()
        self.value.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.target_value.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_1.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_2.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.capital = 10000000
        self.size = 1000
        self.tau = 0.5

    def collect(self, price):

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.new_states = []
        for i in range(0,1000):
            state = []
            # 2. Filter to just the option tickers (exclude the underlying)
            option_ticker_names = [ticker for ticker in self.exchange.market_books if ticker != 'STOCK_UNDERLYING']
            # 3. Pick one randomly
            random_ticker = random.choice(option_ticker_names)

            # 4. Get the order book
            self.orderbook = self.exchange.get_book(random_ticker)

            bids = self.orderbook.get_bids(self.agent_id)
            asks = self.orderbook.get_asks(self.agent_id)

            highest_bid = bids[0][0] if bids else print("no bids") #None
            lowest_ask = asks[0][0] if asks else print("no asks") #None

            volatility = np.random.uniform(low=0.01, high=0.05)

            # ticker_parts = random_ticker.split('_')
            # strike_price = int(ticker_parts[1]) if len(ticker_parts) == 3 else print("no strike") #None
            # spot_price = self.exchange.underlying_price

            # bs = BlaScho()
            # bs.Strike = strike_price
            # bs.Spot = spot_price
            # bs.Volatility =
            # if "CE" in random_ticker:
            #     option_type = "call"
            # elif "PE" in random_ticker:
            #     option_type = "put"
            # else:
            #     option_type = "UNDERLYING"


            state.append(highest_bid)
            state.append(lowest_ask)
            state.append(lowest_ask - highest_bid)
            state.append((lowest_ask + highest_bid)/2)
            state.append(self.capital)
            state.append(self.size)
            #state.append(volatility)

            self.states.append(state)
            # tfstate = tf.convert_to_tensor(state, dtype=tf.float32)
            # action, logprob = self.actor.sample_normal(tfstate, reparameterize=False)

            # tfstate = tf.convert_to_tensor(state, dtype=tf.float32)
            # tfstate = tf.expand_dims(tfstate, axis=0)  # Reshape to (1, 5)

            # Convert state to a 2D tensor
            tfstate = tf.convert_to_tensor(state, dtype=tf.float32)
            #print(f"Before reshape, state shape: {tfstate.shape}")

            # Reshape to (1, 5)
            tfstate = tf.expand_dims(tfstate, axis=0)  # This should result in shape (1, 5)
            #print(f"After reshape, state shape: {tfstate.shape}")

            # Now pass the reshaped state to the actor network
            action, logprob = self.actor.sample_normal(tfstate, reparameterize=False)

            #print("got action")
            self.actions.append(action)
            self.log_probs.append(logprob)
            #print(tf.squeeze(action))
            #print(logprob)
            #print("actions, logprobs appended")


            bid_price = highest_bid + action[0][0] * (lowest_ask - highest_bid) / 2
            ask_price = lowest_ask - action[0][1] * (lowest_ask - highest_bid) / 2
            #print("price done")
            bid_size = tf.math.floor(action[0][2] * 100)
            ask_size = tf.math.floor(action[0][3] * 100)
            #print("size done")
            # bid_price = highest_bid + tf.expand_dims(action[0][0], axis=-1) * (lowest_ask - highest_bid) / 2
            # ask_price = lowest_ask - tf.expand_dims(action[0][1], axis=-1) * (lowest_ask - highest_bid) / 2
            # print("price done")
            # bid_size = tf.math.floor(tf.expand_dims(action[0][2], axis=-1) * 100)
            # ask_size = tf.math.floor(tf.expand_dims(action[0][3], axis=-1) * 100)
            # print("size done")

            print("reqch")
            self.update_book(bid_price.numpy(), ask_price.numpy(), bid_size.numpy(), ask_size.numpy())

            #update orderbook
            reward = (ask_price * ask_size - bid_price * bid_size) - abs(bid_price - highest_bid) - abs(lowest_ask - ask_price)
            if i==500 or i==1500:
              print(action, logprob, reward)
            self.rewards.append(reward)
            #print("rewards appended")
            self.capital += reward
            self.size += bid_size - ask_size

            new_state = []

            bids = self.orderbook.get_bids(self.agent_id)
            asks = self.orderbook.get_asks(self.agent_id)

            highest_bid = bids[0][0] if bids else None
            lowest_ask = asks[0][0] if asks else None


            new_state.append(highest_bid)
            new_state.append(lowest_ask)
            new_state.append(lowest_ask - highest_bid)
            new_state.append((lowest_ask + highest_bid)/2)
            new_state.append(self.capital)
            new_state.append(self.size)
            #new_state.append(volatility + 0.001*np.random.randint(0,10))

            self.new_states.append(new_state)
            price+=10
            #print("end of collect")

    def update_book(self, b_p, a_p, b_s, a_s):

        print("updqtte")

        sidebuy = Side.BUY
        sidesell = Side.SELL

        pricebuy = b_p
        pricesell = a_p

        sizebuy = b_s
        sizesell = a_s

        ordbuy = Order(sidebuy, pricebuy, sizebuy, owner_id=self.agent_id)
        ordsell = Order(sidesell, pricesell, sizesell, owner_id=self.agent_id)

        executed_trades_buy = self.orderbook.add_order(ordbuy)
        executed_trades_sell = self.orderbook.add_order(ordsell)

        if executed_trades_buy:
            print("Order was matched! Trades executed:")
            for trade in executed_trades_buy:
                print(f"Traded {trade.size} at {trade.price} with {trade.maker_id}")
        else:
            print("Order added to the book. No immediate execution.")

        if executed_trades_sell:
            print("Order was matched! Trades executed:")
            for trade in executed_trades_sell:
                print(f"Traded {trade.size} at {trade.price} with {trade.maker_id}")
        else:
            print("Order added to the book. No immediate execution.")


        
    def sample_batch(self):
        states = []
        actions = []
        log_probs = []
        rewards = []
        new_states = []
        for i in range(0,500):
            j =np.random.randint(0,len(self.states))
            states.append(self.states[j])
            actions.append(self.actions[j])
            log_probs.append(self.log_probs[j])
            rewards.append(self.rewards[j])
            new_states.append(self.new_states[j])

        #return states, actions, log_probs, rewards, new_states
        return states, actions, rewards, new_states

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)


    def learn(self):

        self.scale = 2.0
        self.gamma = 0.99
        self.batch_size = 20

        #states, actions, log_probs, rewards, new_states = self.learn()
        for k in range(0,50):
            if k%4==0:
                print(k, end=" ")
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

                value_target = critic_value - log_probs           #tells difference in how good action is and its probability of getting chosen
                                                                #i.e. how far critic and actor are
                value_loss = 0.5 * tf.reduce_mean(tf.square(value, value_target))


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

                actor_loss = log_probs - critic_value                       # -ve of value_target which we want to maximizme and this we want to minimize
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
                critic_1_loss = 0.5 * tf.reduce_mean(tf.square(q1_old_policy, q_hat))
                critic_2_loss = 0.5 * tf.reduce_mean(tf.square(q2_old_policy, q_hat))

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
        print("\n")

    def get_action(self, hb, la):
        self.state = [ hb, la, la - hb, self.capital, self.size]
        state = tf.convert_to_tensor(self.state)
        state = tf.expand_dims(state, axis = 0)
        mu, sigma = self.actor(state)

        bid_price = hb + mu[0][0] * (la - hb) / 2
        ask_price = la - mu[0][1] * (la - hb) / 2
        #print("price done")
        bid_size = tf.math.floor(mu[0][2] * 100)
        ask_size = tf.math.floor(mu[0][3] * 100)

        return bid_price, ask_price, bid_size, ask_size


