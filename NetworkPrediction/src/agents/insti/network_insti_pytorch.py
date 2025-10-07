import torch
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # Use tanh to bound the action between -1 and 1
        action = torch.tanh(self.mu(x))
        return action