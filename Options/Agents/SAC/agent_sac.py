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
        # This is your partner's original logic
        for k in range(10): # simplified loop
            states, actions, rewards, new_states = self.sample_batch()
            # ... The entire learning logic from your partner's file goes here ...
            pass
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