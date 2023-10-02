import torch
from torch import nn as nn
from torch.distributions import Categorical
import os


class Agent(nn.Module):

    def __init__(self, state_size, action_size,num_workers=8,num_steps=128,batch_size=256,lr=0.003,writer=None,device='cpu',
                 learining_rate_decay=0.99,value_loss_coef=0.5,entropy_coef=0.01,clip_grad_norm=0.5,
                 clip_param=0.2, save_path='trained_models/ppo',
                 K_epochs=10):
        super(Agent, self).__init__()
        self.writer = writer
        self.save_path = save_path
        self.device = device
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),

        )
        self.actor = nn.Sequential(
            nn.Linear(256,action_size),
            nn.Softmax(dim=-1)

        )
        self.critic = nn.Sequential(
            nn.Linear(256, 1)
        )
        self.number_epochs =0
        print(self.actor)
        print(self.critic)
        self.K_epochs = K_epochs
        self.num_workers = num_workers
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ortogonal_initialization()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.clip_grad_norm = clip_grad_norm
        self.learning_rate_decay = learining_rate_decay
        self.clip_param = clip_param
        """self.layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )"""

    def ortogonal_initialization(self):

        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        logits = self.actor(x)
        value = self.critic(x)
        dist = Categorical(logits)

        return dist, value

    def get_action(self,obs):
        with torch.no_grad():
            dist, value = self.forward(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)



        return action.detach().numpy(), log_prob.detach().numpy(), value.detach().numpy()


    def decay_learning_rate(self):
        print("Decaying learning rate")
        self.writer.add_scalar("Learning rate", self.optimizer.param_groups[0]['lr'], self.number_epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.learning_rate_decay

    def save_model(self):
        # create folder if not exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        path = self.save_path + str(self.number_epochs) + '.pth'
        torch.save(self.state_dict(), path)
