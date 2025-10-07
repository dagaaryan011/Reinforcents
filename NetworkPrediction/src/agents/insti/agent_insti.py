import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random

# Correctly import from the new network file and model_setup file
from .network_insti_pytorch import CriticNetwork, ActorNetwork
from .model_setup_insti import LongTermModel

# Import other necessary components
from ...tools.buffer import ReplayBuffer
from ...tools.functions import (
    Indicators, macd_signal, get_rsi_conviction,
    get_stoch_conviction, get_current_status, get_ADX,
    get_DMs, get_true_range, get_DX
)
from config import (
    SEQUENCE_LENGTH, AGENT_FPERSONALITIES,
    LONG_TERM_INDICATORS, SHORT_TERM_INDICATORS
)


class Agent_Insti:
    def __init__(self, agent_id: int, alpha: float, beta: float, input_dims: int, tau: float, n_actions: int,
                 rnn_context_model: LongTermModel, gamma: float = 0.99, max_mem_size: int = 1000000, batch_size: int = 64):

        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- RL Networks ---
        self.actor = ActorNetwork(alpha, input_dims, n_actions).to(self.device)
        self.critic = CriticNetwork(beta, input_dims, n_actions).to(self.device)
        self.target_actor = ActorNetwork(alpha, input_dims, n_actions).to(self.device)
        self.target_critic = CriticNetwork(beta, input_dims, n_actions).to(self.device)

        # --- Context Network ---
        self.rnn_context_model = rnn_context_model.to(self.device)

        # RL Memory
        self.memory = ReplayBuffer(max_mem_size, [input_dims], n_actions)
        self.update_network_parameters(tau=1)  # Hard update of target networks at the start

        # --- Agent Personality & State ---
        rand_rsi = np.random.rand(3); self.rsi_coeffs = rand_rsi / np.sum(rand_rsi)
        rand_stoch = np.random.rand(3); self.stoch_coeffs = rand_stoch / np.sum(rand_stoch)

        # State histories and feature windows
        self.long_term_feature_window = deque(maxlen=SEQUENCE_LENGTH)
        self.short_term_feature_window = deque(maxlen=SEQUENCE_LENGTH)
        self.trend, self.macd_history, self.rsi_history, self.k_history, self.d_history = [], [], [], [], []
        self.DMpos_history, self.DMneg_history, self.true_range_history, self.dx_history = [], [], [], []

        print(f"--- Initialized Institutional Agent {self.agent_id} ---")

    def choose_action(self, observation, noise=0.1, evaluate=False):
        self.actor.eval()
        state = torch.tensor(observation, dtype=torch.float).to(self.device)
        mu = self.actor(state).to(self.device)

        if not evaluate:
            mu_prime = mu + torch.tensor(np.random.normal(scale=noise, size=self.actor.n_actions),
                                         dtype=torch.float).to(self.device)
        else:
            mu_prime = mu

        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        new_states = torch.tensor(new_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        self.target_actor.eval(); self.target_critic.eval(); self.critic.train()
        target_actions = self.target_actor(new_states)
        critic_value_ = self.target_critic(new_states, target_actions).view(-1)
        critic_value_[dones] = 0.0
        target = rewards + self.gamma * critic_value_
        
        critic_value = self.critic(states, actions).view(-1)
        
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval(); self.actor.train()
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        
        # Soft update for Actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        # Soft update for Critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def get_rnn_context(self):
        if len(self.long_term_feature_window) < SEQUENCE_LENGTH:
            return None, None

        self.rnn_context_model.eval()
        with torch.no_grad():
            long_term_np = np.array(list(self.long_term_feature_window), dtype=np.float32)
            long_term_tensor = torch.from_numpy(long_term_np).unsqueeze(0).to(self.device)
            long_term_logits = self.rnn_context_model(long_term_tensor)
            long_term_probs = torch.softmax(long_term_logits, dim=1)
            signal = torch.argmax(long_term_probs).item() - 1
            return long_term_probs.cpu().numpy(), signal

    def preprocess_input(self, price: float):
        # 1. Update all raw indicator histories
        self.trend.append(price)
        self.macd_history.append(Indicators.MACD(self.trend))
        self.rsi_history.append(Indicators.Smooth_RSI(self.trend))
        self.k_history.append(Indicators.Stoch_Oscilator(self.trend))
        
        if len(self.k_history) >= 3 and all(k is not None for k in self.k_history[-3:]):
            self.d_history.append(sum(self.k_history[-3:]) / 3)
        else: self.d_history.append(None)
            
        DMpos, DMneg = get_DMs(self.trend)
        self.DMpos_history.append(DMpos); self.DMneg_history.append(DMneg)
        self.true_range_history.append(get_true_range(self.trend))
        self.dx_history.append(get_DX(self.DMpos_history, self.DMneg_history, self.true_range_history))
        
        # 2. Create Long-Term Features and add to window
        long_term_features = self._get_feature_vector(LONG_TERM_INDICATORS)
        self.long_term_feature_window.append(long_term_features)
        
        # 3. Create Short-Term Features and add to window
        short_term_features = self._get_feature_vector(SHORT_TERM_INDICATORS)
        self.short_term_feature_window.append(short_term_features)

    def _get_feature_vector(self, feature_list_to_use: list):
        full_feature_set = {
            'macd': macd_signal(self.macd_history),
            'rsi': get_rsi_conviction(self.rsi_history, coeffs=self.rsi_coeffs),
            'stoch': get_stoch_conviction(self.k_history, self.d_history, coeffs=self.stoch_coeffs),
            'status': get_current_status(self.trend),
            'adx': get_ADX(self.dx_history)
        }
        
        final_features = []
        for indicator_name in feature_list_to_use:
            value = full_feature_set.get(indicator_name, 0.0)
            if value is None or np.isnan(value):
                final_features.append(0.0)
            else:
                final_features.append(value)
        return final_features