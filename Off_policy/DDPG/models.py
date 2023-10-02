import torch
from torch import nn as nn


class Critic(nn.Module):
    def __init__(self,state_size,action_size):
        # DDPG critic network
        super(Critic,self).__init__()
        self.first_layers = nn.Sequential(
                nn.Linear(state_size,400),
                nn.ReLU(),
        )
        self.second_layers = nn.Sequential(
                nn.Linear(400+action_size,300),
                nn.ReLU(),
        )
        self.third_layers = nn.Sequential(
                nn.Linear(300,1),
        )
    def forward(self,state,action):
        x = self.first_layers(state)
        x = self.second_layers(torch.cat([x,action],1))
        x = self.third_layers(x)
        return x


class Actor(nn.Module):
    def __init__(self,state_size,action_size):
        # DDPG actor network
        super(Actor,self).__init__()
        self.layers = nn.Sequential(
                nn.Linear(state_size,400),
                nn.ReLU(),
                nn.Linear(400,300),
                nn.ReLU(),
                nn.Linear(300,action_size),
                nn.Tanh(),
        )

    def forward(self,state):
        x = self.layers(state)

        return x
