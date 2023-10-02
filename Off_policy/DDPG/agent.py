import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from Off_policy.experience_replay import Experience_replay
from Off_policy.DDPG.models import Critic, Actor
from Off_policy.DDPG.utils import Orstein_Uhlenbeck


class Agent(nn.Module):
    def __init__(self,env,writer,discount_factor=0.99,tau=0.001,lr_critic=1e-3,lr_actor=1e-4,
                 replay_memory_size=1000000,replay_memory_sample_size=64,device='cpu',max_steps=1000000,
                 save_path='trained_models/ddpg'):
        super(Agent,self).__init__()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        self.critic = Critic(state_size, action_size)
        self.actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=lr_critic)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=lr_actor)
        self.env = env
        self.memory = Experience_replay(replay_memory_size, replay_memory_sample_size, state_size)
        self.batch_size = replay_memory_sample_size
        self.discount_factor = discount_factor
        self.tau = tau
        self.device = device
        self.writer = writer
        self.max_steps = max_steps
        self.save_path = save_path
        self.total_step = 0


    def act(self,state,deterministic=False):
        with torch.no_grad():

            state = torch.from_numpy(state).float()
            action = self.actor(state)
            action = action.cpu().detach().numpy()
            if deterministic:
                return action
            action += Orstein_Uhlenbeck(action)
        return action # multiply by 2 to get action between -2 and 2

    def train_agent(self):
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


    def save(self):
        filename = self.save_path
        filename += f'{self.env.unwrapped.spec.id}'
        print(f"Saving model to {filename}")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")

    def load(self, filename = "checkpoint"):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))

    def rollout(self):
        total_step = 0
        epoch = 0
        save_freq = 100
        episode = 0
        # logging file

        while total_step < self.max_steps:
            episode_reward = 0
            episode += 1
            state, _ = self.env.reset()

            update_freq = 1
            step = 0
            truncated = False
            while not truncated:
                self.env.render()
                action = self.act(state, deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                truncated = 1 if truncated else 0
                self.memory.add(state, action, reward, next_state, truncated)
                episode_reward += reward
                state = next_state
                if step % update_freq == 0:
                    if self.memory.size >= self.batch_size:
                        self.train_agent()
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
