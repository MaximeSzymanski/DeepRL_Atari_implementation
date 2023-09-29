import numpy as np
import torch
from torch import optim as optim, nn as nn

from DQN.experience_replay import experience_replay
from DQN.model import DQN


class Agent():
    def __init__(self, env, memory_size, sample_size, gamma, epsilon, epsilon_decay, epsilon_min, lr,device,
                 batch_size,writer,model_path='model.pth',
                 seed=0):
        self.writer = writer
        self.model_path = model_path
        self.env = env
        self.number_of_updates = 0
        self.device = device
        self.memory = experience_replay(memory_size, sample_size)
        self.gamma = gamma
        self.seed = np.random.seed(seed)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.network = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_network = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.batch_size = batch_size
    def act(self, state,deterministic=False):
        if deterministic:
            with torch.no_grad():
                return torch.argmax(self.network(state)).item()
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                return self.env.action_space.sample()


            else:
                return torch.argmax(self.network(state)).item()

    def train(self):

            batch_index = np.random.choice((self.memory.size), self.batch_size)
            state, action, reward, next_state, done = self.memory.sample(batch_index)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.from_numpy(action).long().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            done = torch.from_numpy(done).float().to(self.device)
            prediction = self.network(state).gather(1, action)
            self.writer.add_scalar('prediction', prediction.mean(), self.number_of_updates)
            with torch.no_grad():
                target = reward + self.gamma * self.target_network(next_state).detach().max(1)[0].unsqueeze(1) * (1 - done)
            loss = self.loss(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(0.01)

            self.number_of_updates += 1

    def soft_update(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def update_epsilon(self):
        print(f"update epsilon from {self.epsilon} to {max(self.epsilon_min, self.epsilon * self.epsilon_decay)}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self):
        torch.save(self.network.state_dict(), self.model_path)

    def load(self, path='model.pth'):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(torch.load(path, map_location=self.device))

    def play(self, episodes=10):
        for episode in range(episodes):
            state,_ = self.env.reset()
            done = False
            while not done:
                state  = np.array(state)
                action = self.act(torch.from_numpy(state).float().to(self.device),deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = np.array(next_state)

                done = 1 if done else 0
                state = next_state
            print(f'Episode {episode+1} finished with reward {reward}')
        self.env.close()

    def train_agent(self, maximum_number_steps=1e6, save_path='model.pth'):
        total_step = 0
        epoch = 0
        save_freq = 100
        episode = 0
        # logging file

        while total_step < maximum_number_steps:
            episode_reward = 0
            episode += 1
            state,_ = self.env.reset()
            state = np.array(state)
            update_freq = 4
            step = 0
            done = False
            while not done:
                action = self.act(torch.from_numpy(state).float().to(self.device))
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = np.array(next_state)
                self.memory.add(state, action, reward, next_state, done)
                print(f'step : {step}, reward : {reward}, done : {done}')
                done = 1 if done else 0
                episode_reward += reward
                state = next_state
                if step % update_freq == 0:
                    if self.memory.size >= self.batch_size :
                        self.train()
                        # log in logging file


                total_step += 1
                step += 1

            self.writer.add_scalar('reward', episode_reward, episode)
            if episode % 10 == 0:
                print(f'Episode {episode+1} finished, reward : {episode_reward}')
                self.save()
                print(f'epsilon : {self.epsilon}')
                self.update_epsilon()

        self.save()
        self.env.close()
