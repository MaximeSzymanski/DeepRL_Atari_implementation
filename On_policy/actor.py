import torch.nn as nn

class Actor(nn.Module):
    def __init__(self,action_size):
        super(Actor,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        return self.layers(x)