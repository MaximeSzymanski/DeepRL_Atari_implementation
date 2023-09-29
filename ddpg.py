import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
# import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

def Orstein_Uhlenbeck(x,theta=0.15,mu=0,sigma=0.2):
    return theta*(mu-x) + sigma*np.random.randn(1)
class experience_replay():

    # state is a numpy array of shape (memory_size, state_size)


    memory_size = 0
    sample_size = 0

    def __init__(self,memory_size,sample_size,state_size):
        self.memory_size = memory_size
        self.sample_size = sample_size
        shape_state = (memory_size,) + (state_size,)
        shape = (memory_size,1)
        self.state = np.empty(shape_state)
        print(f'state shape: {self.state.shape}')
        self.action = np.empty(shape)
        self.reward = np.empty(shape)
        self.next_state = np.empty(shape_state)
        self.done = np.empty(shape)
        self.head = 0
        self.size = 0


    def add(self,state,action,reward,next_state,done):
        self.state[self.head] = state
        self.action[self.head] = action
        self.reward[self.head] = reward
        self.next_state[self.head] = next_state
        self.done[self.head] = done
        self.head = (self.head + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1


    def sample(self,batch_index):

        return self.state[batch_index],self.action[batch_index],self.reward[batch_index],self.next_state[batch_index],self.done[batch_index]

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



class Agent(nn.Module):
    def __init__(self,env,writer,discount_factor=0.99,tau=0.001,lr_critic=1e-3,lr_actor=1e-4,
                 replay_memory_size=1000000,replay_memory_sample_size=64,device='cpu'):
        super(Agent,self).__init__()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        self.critic = Critic(state_size,action_size)
        self.actor = Actor(state_size,action_size)
        self.target_critic = Critic(state_size,action_size)
        self.target_actor = Actor(state_size,action_size)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=lr_critic)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=lr_actor)
        self.env = env
        self.memory = experience_replay(replay_memory_size,replay_memory_sample_size,state_size)
        self.batch_size = replay_memory_sample_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.device = device
        self.writer = writer
        self.total_step = 0

    def act(self,state,deterministic=False):
        with torch.no_grad():

            state = torch.from_numpy(state).float()
            action = self.actor(state)
            action = action.cpu().detach().numpy()
            if deterministic:
                return action * 2
            action += Orstein_Uhlenbeck(action)
        return action * 2 # multiply by 2 to get action between -2 and 2

    def train(self):
        batch_index = np.random.choice((self.memory.size), self.batch_size)
        state, action, reward, next_state, done = self.memory.sample(batch_index)
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).long().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.from_numpy(done).float().to(self.device)

        y_i = reward + self.target_critic(next_state, self.target_actor(next_state)) * self.discount_factor * (1- done)
        critic_prediction = self.critic(state,action)
        critic_loss = F.mse_loss(critic_prediction,y_i)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        actor_loss = -self.critic(state,self.actor(state)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.writer.add_scalar('loss/actor', actor_loss, self.total_step)
        self.writer.add_scalar('loss/critic', critic_loss, self.total_step)
        self.total_step += 1


        self.update_target_networks()


    def save(self, filename = "checkpoint"):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename = "checkpoint"):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))

    def rollout(self, maximum_number_steps=1e6, save_path='model.pth'):
        total_step = 0
        epoch = 0
        save_freq = 100
        episode = 0
        # logging file

        while total_step < maximum_number_steps:
            episode_reward = 0
            episode += 1
            state, _ = self.env.reset()

            update_freq = 1
            step = 0
            truncated = False
            while not truncated:
                env.render()
                action = self.act(state, deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                truncated = 1 if truncated else 0
                self.memory.add(state, action, reward, next_state, truncated)
                episode_reward += reward
                state = next_state
                if step % update_freq == 0:
                    if self.memory.size >= self.batch_size:
                        self.train()
                #print(f"step {step} finished, reward : {reward}, current episode : {episode}")
                total_step += 1
                step += 1

            self.writer.add_scalar('reward', episode_reward, episode)
            if episode % 10 == 0:
                print(f'Episode {episode + 1} finished, reward : {episode_reward}')
                self.save()


        self.save()


    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        for target_param, param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)







if __name__ == '__main__':
    env = gym.make("Pendulum-v1",render_mode='human')
    writer = SummaryWriter("runs/ddpg")

    print(f'observation space : {env.observation_space.shape[0]}')
    print(f'action space : {env.action_space.shape[0]}')
    agent = Agent(env,writer=writer)
    agent.load()
    agent.rollout()



