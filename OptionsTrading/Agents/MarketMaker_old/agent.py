import numpy as np
import os
from .networks import selector, ValueNetwork, CriticNetwork, ActorNetwork
from .broker import Broker
import tensorflow as tf

class MarketMaker:
    def __init__(self, id):

        self.agent_id = id
        self.broker = Broker()

        self.selector = selector()
        self.actor = ActorNetwork()
        self.critic_1 = CriticNetwork()
        self.critic_2 = CriticNetwork()
        self.value = ValueNetwork()
        self.target_value = ValueNetwork()
        
        self.tau = 0.005
        self.scale = 2.0
        self.gamma = 0.99
        self.alpha = 0.05
        self.epsilon = 1
        self.batch_size = 10
        self.memory_size = 20
        self.batch_times = 5
        self.selectstates = [0] * self.memory_size
        self.states = [0] * self.memory_size
        self.actions = [0] * self.memory_size
        self.log_probs = [0] * self.memory_size
        self.rewards = [0] * self.memory_size
        self.new_states = [0] * self.memory_size

        self.t = 0

    def load_models(self):
        agent_file_directory = os.path.dirname(os.path.abspath(__file__))
        os.path.join(agent_file_directory, 'models')
        selector = os.path.join(agent_file_directory, f'selector_{self.agent_id}')
        target = os.path.join(agent_file_directory, f'target_{self.agent_id}')
        target_value = os.path.join(agent_file_directory, f'target_value_{self.agent_id}')
        critic_1 = os.path.join(agent_file_directory, f'critic_1_{self.agent_id}')
        critic_2 = os.path.join(agent_file_directory, f'critic_2_{self.agent_id}')
        actor = os.path.join(agent_file_directory, f'actor_{self.agent_id}')


    
    def save_models(self):


    def collect(self):

        i = self.t % self.memory_size

        self.broker.get_notifications(self.agent_id)

        allstates = self.broker.get_all_states()   # gets state of all orderbooks together for sleecting
        self.selectstates[i] =allstates
        selection = self.select(allstates)
        print(selection)

        random = np.random.rand()    # greedy
        if random < self.epsilon:
            idx = np.random.randint(0,24)
        else :
            idx = np.argmax(selection)

        print(idx)
        ticker = self.broker.env.tickers_list[idx]   # get the chosen ticker

        state = self.broker.get_actual_state(ticker)
        action, log_probs = self.get_action(state)
        bid_price, bid_size, ask_price, ask_size = self.decide_values(action, state[1], state[2])  # decide the price and sizes

        self.broker.update_book(ticker, bid_price, bid_size, ask_price, ask_size, self.agent_id)   # place the order

        reward = self.get_reward(ticker, bid_price, bid_size, ask_price, ask_size, state[1], state[2])

        self.states[i] = state
        self.actions[i] = action
        self.log_probs[i] = log_probs
        self.rewards[i] = reward
        print(state)
        print(reward)

        if i-10>=0:
            self.new_states[i-10] = state
        else:
            self.new_states[self.memory_size+i-10] = state   #new state  will be state after 10 steps

        self.t += 1

    def select(self, allstates):
        a = []

        for state in allstates:
            state_tensor = tf.expand_dims(state, axis=0) # Shape (1, 14)
            a.append(state_tensor)

        selection = self.selector(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], 
                                  a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23])
        
        return selection.numpy()   # returns numpy of selectionn values
         
    def get_action(self, state):
        tfstate = tf.convert_to_tensor([state], dtype=tf.float32)
        action, logprob = self.actor.sample_normal(tfstate, reparameterize=False)
        return action.numpy(), logprob.numpy()
    
    def decide_values(self, action, highest_bid, lowest_ask):
        action_values = action[0]
        print(action_values)
        size = min(self.broker.temp_inventory, 10)
        price = min((lowest_ask - highest_bid)/2, self.broker.temp_capital)
        bid_price = highest_bid + action_values[0] * price
        ask_price = lowest_ask - action_values[1] * (lowest_ask - highest_bid) / 2
        bid_size = int(action_values[2] * 10)
        ask_size = int(action_values[3] * size)
        print(bid_price, ask_price, bid_size, ask_size)
        return bid_price, ask_price, bid_size, ask_size
        
    
    def get_reward(self, ticker, bid_price, bid_size, ask_price, ask_size, highest_bid, lowest_ask):
        PL = ask_price * ask_size - bid_price * bid_size
        diff_bid = abs(bid_price - highest_bid)
        diff_ask = abs(lowest_ask - ask_price)
        reward = PL - diff_bid - diff_ask
        size_diff = bid_size - ask_size
        self.broker.temp_capital+=PL
        self.broker.temp_inventory += size_diff
        return reward
    
    def assign_settlements(self, final):
        self.broker.settlement(final)
        self.expiry_rewards = []
        for ticker in self.broker.env.tickers_list:
            print(self.broker.portfolio[ticker])
            print(self.broker.cash_settlement[ticker])
            self.expiry_rewards.append(self.broker.cash_settlement[ticker])    # at the end get the sttlement calculated for each ticker (comparing selector values with this)
    
    def assign_volumes(self):
        self.expiry_volumes = []
        for ticker in self.broker.env.tickers_list:
            self.expiry_volumes.append(self.broker.env.total_volumes[ticker])

    def sample_batch(self):
        allstates, states, actions, log_probs, rewards, new_states = [], [], [], [], [], []    
        max_mem = min(self.t, self.memory_size)
        for _ in range(self.batch_size):
            j = np.random.randint(0, max_mem)
            allstates.append(self.selectstates[j])
            states.append(self.states[j])
            actions.append(self.actions[j])
            log_probs.append(self.log_probs[j])
            rewards.append(self.rewards[j])
            new_states.append(self.new_states[j])

        return allstates, states, actions, log_probs, rewards, new_states


    def update_network_parameters(self):
        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * self.tau + targets[i] * (1 - self.tau))
        self.target_value.set_weights(weights)

    def learn(self):

        for k in range(0,self.batch_times):
            
            all_states_list_of_samples, states, actions, log_probs, rewards, new_states = self.sample_batch()

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            actions = tf.squeeze(actions, 1)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            log_probs = tf.convert_to_tensor(log_probs, dtype=tf.float32)
            expiry_volumes = tf.convert_to_tensor(self.expiry_volumes, dtype=tf.float32)


            all_states_tensor = tf.convert_to_tensor(all_states_list_of_samples, dtype=tf.float32) 
            t = tf.unstack(all_states_tensor, axis=1) 


            # SELECTOR UPDATE
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(self.selector(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], 
                                                              t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18],
                                                               t[19], t[20], t[21], t[22], t[23]) - expiry_volumes))
            selector_gradient = tape.gradient(loss, self.selector.trainable_variables)
            self.selector.optimizer.apply_gradients(zip(selector_gradient, self.selector.trainable_variables))


            # VALUE UPDATE
            with tf.GradientTape() as tape:
                value = tf.squeeze(self.value(states), 1)
                value_ = tf.squeeze(self.target_value(states_), 1)

                current_policy_actions, current_log_probs = self.actor.sample_normal(states, reparameterize=False)
                q1_new_policy = self.critic_1(states, current_policy_actions)
                q2_new_policy = self.critic_2(states, current_policy_actions)
                critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
                
                # Squeeze log_probs to shape (BatchSize,)
                current_log_probs_squeezed = tf.squeeze(current_log_probs, 1)

                value_target = critic_value - self.alpha * current_log_probs_squeezed
                
                value_loss = 0.5 * tf.reduce_mean(tf.square(value - value_target))

            value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
            self.value.optimizer.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))


            # ACTOR UPDATE
            with tf.GradientTape() as tape:
                new_policy_actions, new_policy_log_probs = self.actor.sample_normal(states, reparameterize=True)
                
                # Squeeze log_probs to shape (BatchSize,)
                new_policy_log_probs_squeezed = tf.squeeze(new_policy_log_probs, 1)
                
                q1_new_policy = self.critic_1(states, new_policy_actions)
                q2_new_policy = self.critic_2(states, new_policy_actions)
                critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)

                # Actor Loss: alpha * log_pi - Q_value
                actor_loss = self.alpha * new_policy_log_probs_squeezed - critic_value
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))


            # CRITICS UPDATE
            with tf.GradientTape(persistent=True) as tape:
                self.scale=0.5
                dones = 0
                q_hat = self.scale*rewards + self.gamma*value_*(1-dones)
                q1_old_policy = tf.squeeze(self.critic_1(states, actions), 1)
                q2_old_policy = tf.squeeze(self.critic_2(states, actions), 1)
                critic_1_loss = 0.5 * tf.reduce_mean(tf.square(q1_old_policy - q_hat))
                critic_2_loss = 0.5 * tf.reduce_mean(tf.square(q2_old_policy - q_hat))

            critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
            critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

            self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))
            self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))

            
            # TARGET UPDATE
            self.update_network_parameters()

        self.expiry_rewards = []
        self.epsilon *= 0.999999
        self.expiry_volumes = []

    def action_at_expiry(self, initial, final):
        self.assign_settlements(final)
        self.assign_volumes()
        self.broker.settle()
        PL = self.broker.get_PL(initial, final)
        print(f"{self.agent_id} : {PL}")
        self.learn()
        self.broker.new_option()     # resets for new option trading 
        self.broker.reset_portfolio()   # empties the portfolio for tickers of next options trading

def initialize_MM_agent(agent_id):
    Agent = MarketMaker(agent_id)
    return Agent


