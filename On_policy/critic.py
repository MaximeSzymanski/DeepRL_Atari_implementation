import torch.nn as nn

class Critic(nn.Module):
    def __init__(self,hidden_size):
        super(Critic,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
    def forward(self,x):
        return self.layers(x)