import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np


class Selector(nn.Module):
    def __init__(self):
        super(Selector, self).__init__()
        
        self.indiv1 = nn.Linear(14, 16)
        self.indiv2 = nn.Linear(16, 8)
        self.indiv3 = nn.Linear(8, 4)
        self.indiv4 = nn.Linear(4, 2)
        self.indiv5 = nn.Linear(2, 1)

        self.choose1 = nn.Linear(24, 24)
        self.choose2 = nn.Linear(24, 24)
        self.choose3 = nn.Linear(24, 24)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state1, state2, state3, state4, state5, 
               state6, state7, state8, state9, state10, 
               state11, state12, state13, state14, state15, 
               state16, state17, state18, state19, state20,
               state21, state22, state23, state24):
        
        states = []
        for state in [state1, state2, state3, state4, state5,
                    state6, state7, state8, state9, state10,
                    state11, state12, state13, state14, state15,
                    state16, state17, state18, state19, state20,
                    state21, state22, state23, state24]:
            state = F.leaky_relu(self.indiv1(state), negative_slope=0.01)
            state = F.leaky_relu(self.indiv2(state), negative_slope=0.01)
            state = F.leaky_relu(self.indiv3(state), negative_slope=0.01)
            state = F.leaky_relu(self.indiv4(state), negative_slope=0.01)
            state = F.leaky_relu(self.indiv5(state), negative_slope=0.01)
            states.append(state)

        state = torch.cat(states, dim=-1)

        state = F.leaky_relu(self.choose1(state), negative_slope=0.01)
        state = F.leaky_relu(self.choose2(state), negative_slope=0.01)
        state = self.choose3(state)

        return state


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(14 + 4, 32) 
        self.fc2 = nn.Linear(32, 32)
        self.q = nn.Linear(32, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        # Input size is state_size (15)
        self.fc1 = nn.Linear(14, 32)
        self.fc2 = nn.Linear(32, 32)
        self.v = nn.Linear(32, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)


class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.n_actions = 4
        
        self.fc1 = nn.Linear(14, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8) 
        
        self.alpha_layer = nn.Linear(8, self.n_actions)
        self.beta_layer = nn.Linear(8, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        alpha = self.alpha_layer(x)
        beta = self.beta_layer(x)
        
        return alpha, beta        # learned for beta distribution

    def sample_normal(self, state, reparameterize=True):
        alpha, beta = self.forward(state)
        
        alpha = F.softplus(alpha) + 1e-6
        beta = F.softplus(beta) + 1e-6
        
        probabilities = Beta(alpha, beta) 
        
        action = probabilities.sample()
        
        log_probs = probabilities.log_prob(action)
        
        log_probs = torch.logsumexp(log_probs, dim=1, keepdim=True)
        
        return action, log_probs