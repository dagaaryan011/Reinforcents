import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np
import os
from .networks import Selector, ActorNetwork, CriticNetwork, ValueNetwork
from .broker import Broker

class MarketMaker:
    def __init__(self, id):

        self.agent_id = id
        self.broker = Broker()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selector = Selector().to(self.device)
        self.actor = ActorNetwork().to(self.device)
        self.critic_1 = CriticNetwork().to(self.device)
        self.critic_2 = CriticNetwork().to(self.device)
        self.value = ValueNetwork().to(self.device)
        self.target_value = ValueNetwork().to(self.device)
        
        self.tau = 0.005
        self.scale = 2.0
        self.gamma = 0.99
        self.alpha = 0.05
        self.epsilon = 1
        self.batch_size = 50
        self.memory_size = 200
        self.batch_times = 10
        self.selectstates = [0] * self.memory_size
        self.states = [0] * self.memory_size
        self.actions = [0] * self.memory_size
        self.rewards = [0] * self.memory_size
        self.new_states = [0] * self.memory_size

        self.expiry_volumes = []
        self.expiry_volumes_highest_index = 0

        self.t = 0#timestamp for learning

    def save_models(self):

        agent_file_directory = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(agent_file_directory, 'models')
        os.makedirs(model_dir, exist_ok=True) 

        def save_network(network, optimizer, name):
            model_path = os.path.join(model_dir, f'{name}_{self.agent_id}.pt')
            torch.save(network.state_dict(), model_path)
            
            if optimizer:
                optim_path = os.path.join(model_dir, f'{name}_optim_{self.agent_id}.pt')
                torch.save(optimizer.state_dict(), optim_path)

        # print("\nSaving models...")
        save_network(self.selector, self.selector.optimizer, 'selector')
        save_network(self.actor, self.actor.optimizer, 'actor')
        save_network(self.critic_1, self.critic_1.optimizer, 'critic_1')
        save_network(self.critic_2, self.critic_2.optimizer, 'critic_2')
        save_network(self.value, self.value.optimizer, 'value')
        save_network(self.target_value, None, 'target_value') # No optimizer
        # print("Save complete.")


    def load_models(self):
        print("\nAttempting to load models...")

        # 1. Define the simple base path string for your models
        #    This is the only line you'll need to change in the future.
        model_dir = r"D:\NetworkPrediction\data\models\MarketMaker"

        # 2. A dictionary mapping the model names to the actual network objects
        models_to_load = {
            'selector': self.selector,
            'actor': self.actor,
            'critic_1': self.critic_1,
            'critic_2': self.critic_2,
            'value': self.value,
            'target_value': self.target_value,
            'selector_optim' : self.selector.optimizer,
            'actor_optim' : self.actor.optimizer,
            'critic_1_optim' : self.critic_1.optimizer,
            'critic_2_optim' : self.critic_2.optimizer,
            'value_optim' : self.value.optimizer
        }

        # 3. Loop through and load each model
        for name, network in models_to_load.items():
            # Construct the simple, full path string for the model
            path = f"{model_dir}\\{name}_MM_{1}.pt"

            try:
                if os.path.isfile(path):
                    network.load_state_dict(torch.load(path, map_location=self.device))
                    print(f"Successfully loaded {name} from: {path}")
                else:
                    print(f"File not found for {name}: {path}")
            except Exception as e:
                print(f"Error loading {name} model: {e}")

    def collect(self):


        i = self.t % self.memory_size

        self.broker.get_notifications(self.agent_id)

        allstates = self.broker.get_all_states()   # gets state of all orderbooks together for sleecting
        self.selectstates[i] =allstates
        selection = self.select(allstates)
        probabilities = F.softmax(selection, dim=1).squeeze(0).detach().numpy()
        #print(selection)
        self.expiry_volumes_highest_index = self.assign_volumes()
        
        idx = np.random.choice(24, p=probabilities)

        #print(idx)
        ticker = self.broker.env.tickers_list[idx]   # get the chosen ticker
        print(f"{self.agent_id} putting order in {ticker} i.e. {idx}")
        state = self.broker.get_actual_state(ticker)
        action, log_probs = self.get_action(state)
        bid_price, bid_size, ask_price, ask_size = self.decide_values(action, state[1], state[2])  # decide the price and sizes

        self.broker.update_book(ticker, bid_price, bid_size, ask_price, ask_size, self.agent_id)   # place the order

        reward = self.get_reward(ticker, bid_price, bid_size, ask_price, ask_size, state[1], state[2])

        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        # print(state)
        #print(reward)

        if i-10>=0:
            self.new_states[i-10] = state
        else:
            self.new_states[self.memory_size+i-10] = state   #new state  will be state after 10 steps

        self.t += 1

        if self.t % 200 == 0 and self.t > 0 :
            self.expiry_volumes_highest_index = self.assign_volumes()
            self.learn()

        if self.t == 1000000:
            self.t = 0

        

    def select(self, allstates):
        a = []

        for state in allstates:
            # Convert each state (list/np.array of shape (15,)) to tensor (1, 15)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            a.append(state_tensor)

        selection = self.selector(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], 
                                  a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21], a[22], a[23])
        
        return selection   # returns numpy of selectionn values
         
    def get_action(self, state):
        tfstate =  torch.tensor([state], dtype=torch.float32, device=self.device)
        action, logprob = self.actor.sample_normal(tfstate, reparameterize=False)
        return action.detach().numpy(), logprob.detach().numpy()
    
    def decide_values(self, action, highest_bid, lowest_ask):
        
        action_values = action[0]
        # print(action_values)
        size = min(self.broker.temp_inventory, 10)
        price = min((lowest_ask - highest_bid)/2, self.broker.temp_capital)
        bid_price = highest_bid + action_values[0] * price
        ask_price = lowest_ask - action_values[1] * (lowest_ask - highest_bid) / 2
        bid_size = int(action_values[2] * 10)
        ask_size = int(action_values[3] * size)
        # print(bid_price, ask_price, bid_size, ask_size)
        return bid_price, ask_price, bid_size, ask_size
        
    
    def get_reward(self, ticker, bid_price, bid_size, ask_price, ask_size, highest_bid, lowest_ask):
        #good  , no changes here
        PL = ask_price * ask_size - bid_price * bid_size
        diff_bid = abs(bid_price - highest_bid)
        diff_ask = abs(lowest_ask - ask_price)
        reward = PL - diff_bid - diff_ask
        size_diff = bid_size - ask_size
        self.broker.temp_capital+=PL
        self.broker.temp_inventory += size_diff
        return reward
    
    # def assign_settlements(self, final):
    #     self.broker.settlement(final)
    #     self.expiry_rewards = []
    #     for ticker in self.broker.env.tickers_list:
    #         # print(self.broker.portfolio[ticker])
    #         # print(self.broker.cash_settlement[ticker])
    #         self.expiry_rewards.append(self.broker.cash_settlement[ticker])    # at the end get the sttlement calculated for each ticker (comparing selector values with this)
    
    def assign_volumes(self):
        expiry_volumes = []
        for ticker in self.broker.env.tickers_list:
            
            #print(self.broker.env.total_volumes[ticker])
            expiry_volumes.append(self.broker.env.total_volumes[ticker])
        
        expiry_volumes = np.array(expiry_volumes)

        idx = np.argmax(expiry_volumes)

        return idx
        # expiry_volumes = np.array(expiry_volumes)
        # idx = np.argmax(expiry_volumes)
        # for i in range(len(expiry_volumes)):
        #     if i == idx :
        #         expiry_volumes[i] = 1
        #     else:
        #         expiry_volumes[i] = 0

        # return expiry_volumes

    def sample_batch(self):
        allstates, states, actions, rewards, new_states = [], [], [], [], []
        max_mem = min(self.t, self.memory_size)
        for _ in range(self.batch_size):
            j = np.random.randint(0, max_mem)
            allstates.append(self.selectstates[j])
            states.append(self.states[j])
            actions.append(self.actions[j])
            rewards.append(self.rewards[j])
            new_states.append(self.new_states[j])

        return allstates, states, actions, rewards, new_states


    def update_network_parameters(self):
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = self.tau*value_state_dict[name].clone() + (1-self.tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def learn(self):

        for k in range(0,self.batch_times):
            
            all_states_list_of_samples, states, actions, rewards, new_states = self.sample_batch()

            states = torch.tensor(states, dtype=torch.float, device = self.device)
            states_ = torch.tensor(new_states, dtype=torch.float, device=self.device)
            actions = torch.tensor(actions, dtype=torch.float, device = self.device)
            actions = torch.squeeze(actions, 1)
            rewards = torch.tensor(rewards, dtype=torch.float, device = self.device)

            target_index = self.expiry_volumes_highest_index
            target_labels = torch.full((self.batch_size,), target_index, dtype=torch.long, device = self.device)

            all_states_tensor = torch.tensor(all_states_list_of_samples, dtype=torch.float, device = self.device) 
            t = torch.unbind(all_states_tensor, axis=1) 

            
            logits = self.selector(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], 
                       t[10], t[11], t[12], t[13], t[14], t[15], t[16], t[17], t[18],
                       t[19], t[20], t[21], t[22], t[23])


            # SELECTOR UPDATE
            self.selector.optimizer.zero_grad()
            selector_loss = F.cross_entropy(logits, target_labels)
            selector_loss.backward(retain_graph=True)
            self.selector.optimizer.step()

            value = self.value(states).view(-1)
            value_ = self.target_value(states_).view(-1)
            

            new_actions, new_log_probs = self.actor.sample_normal(states, reparameterize=False)
            new_log_probs = new_log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(states, new_actions)
            q2_new_policy = self.critic_2.forward(states, new_actions)
            critic_value = torch.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)

            # VALUE UPDATE
            self.value.optimizer.zero_grad()
            value_target = critic_value - self.alpha * new_log_probs
            value_loss = 0.5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            new_actions, new_log_probs = self.actor.sample_normal(states, reparameterize=True)
            new_log_probs = new_log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(states, new_actions)
            q2_new_policy = self.critic_2.forward(states, new_actions)
            critic_value = torch.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)
            
            # ACTOR UPDATE
            actor_loss = self.alpha * new_log_probs - critic_value
            actor_loss = torch.mean(actor_loss)
            self.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor.optimizer.step()

            # CRITIC UPDATE
            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            q_hat = self.scale*rewards + self.gamma*value_
            q1_old_policy = self.critic_1.forward(states, actions).view(-1)
            q2_old_policy = self.critic_2.forward(states, actions).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

            critic_loss = critic_1_loss + critic_2_loss
            critic_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()

            self.update_network_parameters()

            self.save_models()

        self.expiry_rewards = []
        self.epsilon *= 0.999
        self.expiry_volumes = []

    def action_at_expiry(self, initial, final):
        #self.assign_settlements(final)
        self.broker.settlement(final)
        self.expiry_volumes_highest_index = self.assign_volumes()
        self.broker.settle()
        PL = self.broker.get_PL(initial, final)
        print("final ", final)
        print(f"{self.agent_id} PL: {PL}")
        print(f"{self.agent_id} capital: {self.broker.capital}")
        print(f"{self.agent_id} inventory: {self.broker.inventory}")
        self.learn()
        self.broker.new_day()     # resets for new option trading 
        self.broker.reset_portfolio()   # empties the portfolio for tickers of next options trading


def initialize_MM_agent(agent_id):
    Agent = MarketMaker(agent_id)
    return Agent

