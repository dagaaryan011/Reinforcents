from .broker import Broker
from .networks import Selector, Q
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta

class RetailTrader:
    def __init__(self, id):
        self.agent_id = id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.broker = Broker()
        self.call_selector = Selector().to(self.device)
        self.put_selector = Selector().to(self.device)
        self.q = Q().to(self.device)
        self.gamma = 0.9
        self.epsilon = 1

        self.memory_size = 100
        self.batch_size = 50
        self.batch_times = 10
        self.states = [0] * self.memory_size
        self.rewards = [0] * self.memory_size
        self.new_states = [0] * self.memory_size
        self.select_states_call = [0] * self.memory_size
        self.select_states_put = [0] * self.memory_size
        self.max_settle_idx_call = 0
        self.max_settle_idx_put = 0
        self.t = 0 

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

        print("\nSaving models...")
        save_network(self.call_selector, self.call_selector.optimizer, 'call_selector')
        save_network(self.put_selector, self.put_selector.optimizer, 'put_selector')
        save_network(self.q, self.q.optimizer, 'q')
        print("Save complete.")


    def load_models(self):
        
        agent_file_directory = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(agent_file_directory, 'models')
        
        def load_network(network, optimizer, name):
            # 1. Load Model Weights
            model_path = os.path.join(model_dir, f'{name}_{self.agent_id}.pt')
            
            if os.path.exists(model_path):
                try:
                    network.load_state_dict(torch.load(model_path, map_location=self.device)) 
                    print(f"Loaded {name} weights.")
                except Exception as e:
                    print(f"Warning: Failed to load {name}. {e}")
            else:
                print(f"File not found: {model_path}. ")

            # 2. Load Optimizer State
            if optimizer:
                optim_path = os.path.join(model_dir, f'{name}_optim_{self.agent_id}.pt')
                if os.path.exists(optim_path):
                    try:
                        # 🔑 Correctly load into the optimizer object
                        optimizer.load_state_dict(torch.load(optim_path, map_location=self.device))
                        print(f"Loaded {name} optimizer state.")
                    except Exception as e:
                        print(f"Failed to load {name} optimizer.{e}")
                else:
                    print(f"Optimizer state not found for {name}. ")
            
        print("\nAttempting to load models...")
        load_network(self.call_selector, self.call_selector.optimizer, 'call_selector')
        load_network(self.put_selector, self.put_selector.optimizer, 'put_selector')
        load_network(self.q, self.q.optimizer, 'q')

    def collect(self):

        i = self.t % self.memory_size

        trend_state = self.broker.get_actual_trend_state()
        self.states[i] = trend_state

        actions = self.get_action(trend_state)
        print("retail actions\n", actions)
        random = np.random.rand()# greedy
        if random < self.epsilon:
            action = np.random.randint(0,5)
        else :
            action = np.argmax(actions)

        print("retail action", action)
        
        if action==0 or action==1:   # buy call , sell call   so selector for call tickers

            
            if action == 0 and self.broker.capital <= 500 :
                reward = -10
            
            elif action == 1 and self.broker.inventory <= 10 :
                reward = -10
            
            else:
                allstates = self.broker.get_all_states_calls()   # gets state of all orderbooks together for sleecting
                self.select_states_call[i] =allstates
                selection = self.select(allstates, self.call_selector)
                probabilities = F.softmax(selection, dim=1).squeeze(0).detach().numpy()
                # print(selection)

                idx = np.random.choice(12, p=probabilities)
                ticker = self.broker.env.call_list[idx]
                self.broker.update_book(action, ticker, self.agent_id)

                reward = self.get_reward(action, ticker)
            self.rewards[i] = reward

        elif action==2 or action==3:

            if action == 0 and self.broker.capital <= 500 :
                reward = -10
            
            elif action == 1 and self.broker.inventory <= 10 :
                reward = -10

            else:
                allstates = self.broker.get_all_states_puts()   # gets state of all orderbooks together for sleecting
                self.select_states_put[i] =allstates
                selection = self.select(allstates, self.put_selector)
                probabilities = F.softmax(selection, dim=1).squeeze(0).detach().numpy()
                # print(selection)

                idx = np.random.choice(12, p=probabilities)
                ticker = self.broker.env.put_list[idx]
                self.broker.update_book(action, ticker, self.agent_id)

                reward = self.get_reward(action, ticker)

            self.rewards[i] = reward

        else:
            reward = 0.001
            self.rewards[i] = reward

        if i-10>=0:
            self.new_states[i-10] = trend_state
        else:
            self.new_states[self.memory_size+i-10] = trend_state   #new state  will be state after 10 steps

        self.t += 1
   
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float, device = self.device)
        return self.q.forward(state).detach().numpy()
    
    def select(self, allstates, selector):
        a = []

        for state in allstates:
            # Convert each state (list/np.array of shape (15,)) to tensor (1, 15)
            state_tensor = torch.tensor(state, dtype=torch.float32, device = self.device).unsqueeze(0)
            a.append(state_tensor)

        selection = selector(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11])
        
        return selection 

    def get_reward(self, action, ticker):
        parts = ticker.split('_')
        strike = int(parts[1])
        if action==0 or action==2:
            return strike - self.broker.env.trend[-1]
        else:
            return self.broker.env.trend[-1] - strike


    def assign_settlements(self , final):
        self.broker.settlement(final)
        call_settle = []
        put_settle = []
        for ticker in self.broker.env.call_list:
            # print(self.broker.portfolio[ticker])
            # print(self.broker.cash_settlement[ticker])
            call_settle.append(self.broker.cash_settlement[ticker]) 
        for ticker in self.broker.env.put_list:
            # print(self.broker.portfolio[ticker])
            # print(self.broker.cash_settlement[ticker])
            put_settle.append(self.broker.cash_settlement[ticker]) 

        self.max_settle_idx_call = np.argmax(np.array(call_settle))
        self.max_settle_idx_put = np.argmax(np.array(put_settle))

        

    def sample_batch(self):
        states, new_states, rewards, select_states_call, select_states_put = [], [], [], [], []
        max_mem = min(self.t, self.memory_size)
        for _ in range(self.batch_size):
            j = np.random.randint(0, max_mem)
            states.append(self.states[j])
            new_states.append(self.new_states[j])
            rewards.append(self.rewards[j])
            select_states_call.append(self.select_states_call[j])
            select_states_put.append(self.select_states_put[j])

        return states, new_states, rewards, select_states_call, select_states_put

    
    def learn(self):

        for i in range(0,self.batch_times):
            states , new_states, rewards, select_states_call, select_states_put = self.sample_batch()

            states = torch.tensor(states, dtype=torch.float, device = self.device)
            new_states = torch.tensor(new_states, dtype=torch.float, device = self.device)
            rewards = torch.tensor(rewards, dtype=torch.float, device = self.device).unsqueeze(1)

            select_states_call = torch.tensor(select_states_call, dtype=torch.float, device = self.device)
            s = torch.unbind(select_states_call, axis = 1)
            logits = self.call_selector(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11])
            self.call_selector.optimizer.zero_grad()
            call_target = torch.full((self.batch_size,), self.max_settle_idx_call, dtype=torch.long, device = self.device)
            call_selector_loss = F.cross_entropy(logits, call_target)
            call_selector_loss.backward(retain_graph=True)
            self.call_selector.optimizer.step()

            select_states_put = torch.tensor(select_states_put, dtype=torch.float, device = self.device)
            s = torch.unbind(select_states_put, axis = 1)
            logits = self.put_selector(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11])
            self.put_selector.optimizer.zero_grad()
            put_target = torch.full((self.batch_size,), self.max_settle_idx_put, dtype=torch.long, device = self.device)
            put_selector_loss = F.cross_entropy(logits, put_target)
            put_selector_loss.backward(retain_graph=True)
            self.put_selector.optimizer.step()

            self.q.optimizer.zero_grad()
            q = self.q(states)
            q_new = self.q(new_states)
            q_target = rewards + self.gamma * torch.max(q_new, dim=1)[0].unsqueeze(1)
            q_loss = F.mse_loss(q_target, q)
            q_loss.backward()
            self.q.optimizer.step()

            self.save_models()



    def action_at_expiry(self, initial, final):
        self.assign_settlements(final)
        self.broker.settle()
        PL = self.broker.get_PL(initial, final)
        # print(f"{self.agent_id} : {PL}")
        self.learn()
        self.broker.new_day()     # resets for new option trading 
        self.broker.reset_portfolio()   # empties the portfolio for tickers of next options trading

def initialize_RT_agent(agent_id):
    Agent = RetailTrader(agent_id)
    return Agent