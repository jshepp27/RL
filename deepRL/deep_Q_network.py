import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearDeepNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        # Predict Value of an Action
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions