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

        self.choose1 = nn.Linear(12, 12)
        self.choose2 = nn.Linear(12, 12)
        self.choose3 = nn.Linear(12, 12)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state1, state2, state3, state4, state5, 
               state6, state7, state8, state9, state10, 
               state11, state12):
        
        states = []
        for state in [state1, state2, state3, state4, state5,
                    state6, state7, state8, state9, state10,
                    state11, state12]:
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


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()

        self.fc1 = nn.Linear(8, 16) 
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16, 8)
        self.q = nn.Linear(8,5)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.q(x))
        return x