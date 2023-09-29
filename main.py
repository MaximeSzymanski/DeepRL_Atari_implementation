import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import stable_baselines3
import numpy as np
import gym
import tqdm
import os
class experience_replay():

    # state is a numpy array of shape (memory_size, state_size)


    memory_size = 0
    sample_size = 0

    def __init__(self,memory_size,sample_size,state_size=(4,84,84)):
        self.memory_size = memory_size
        self.sample_size = sample_size
        shape_state = (memory_size,) + state_size
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


class DQN(nn.Module):

    def __init__(self,input_size, output_size):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        """self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )"""

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.layers(x)

class Agent():
    def __init__(self, env, memory_size, sample_size, gamma, epsilon, epsilon_decay, epsilon_min, lr,seed=0):
        self.env = env
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = experience_replay(memory_size, sample_size)
        self.gamma = gamma
        self.seed = np.random.seed(seed)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.network = DQN( env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_network = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def act(self, state,deterministic=False):
        print(f"state shape : {state.shape}")
        if deterministic:
            with torch.no_grad():
                return torch.argmax(self.network(state)).item()
        with torch.no_grad():
            if np.random.rand() < self.epsilon:
                return self.env.action_space.sample()


            else:
                return torch.argmax(self.network(state)).item()

    def train(self, batch_size):


            batch_index = np.random.choice((self.memory.size), batch_size)
            state, action, reward, next_state, done = self.memory.sample(batch_index)
            state = torch.from_numpy(state).float().to(self.device)
            action = torch.from_numpy(action).long().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            done = torch.from_numpy(done).float().to(self.device)
            prediction = self.network(state).gather(1, action)
            with torch.no_grad():
                target = reward + self.gamma * self.target_network(next_state).detach().max(1)[0].unsqueeze(1) * (1 - done)
            loss = self.loss(prediction, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(0.01)

    def soft_update(self, tau):
        for target_param, local_param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def update_epsilon(self):
        print(f"update epsilon from {self.epsilon} to {max(self.epsilon_min, self.epsilon * self.epsilon_decay)}")
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path='model.pth'):
        torch.save(self.network.state_dict(), path)

    def load(self, path='model.pth'):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.target_network.load_state_dict(torch.load(path, map_location=self.device))

    def play(self, episodes=10):
        for episode in range(episodes):
            state,_ = self.env.reset()
            done = False
            while not done:
                self.env.render()
                state  = np.array(state)
                action = self.act(torch.from_numpy(state).float().to(self.device),deterministic=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                next_state = np.array(next_state)

                done = 1 if done else 0
                state = next_state
            print(f'Episode {episode+1} finished with reward {reward}')
        self.env.close()

    def train_agent(self, episodes=1000, batch_size=32, update_target_network=100, save_path='model.pth'):
        total_step = 0
        epoch = 0
        save_freq = 100
        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('episode,timestep,reward\n')

        log_freq = 10
        log_running_reward = 0
        log_running_episodes = 0
        for episode in range(episodes):
            episode_reward = 0
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
                    if self.memory.size >= batch_size :
                        self.train(batch_size)
                        # log in logging file
                if total_step % log_freq == 0 :
                            # log average reward till last episode
                            if log_running_episodes == 0:
                                log_running_episodes = 1
                            log_avg_reward = log_running_reward / log_running_episodes
                            log_avg_reward = round(log_avg_reward, 4)

                            log_f.write('{},{},{}\n'.format(episode, total_step, log_avg_reward))
                            log_f.flush()

                            log_running_reward = 0
                            log_running_episodes = 0

                total_step += 1
                step += 1

            log_running_reward += episode_reward
            log_running_episodes += 1
            if episode % 10 == 0:
                print(f'Episode {episode+1} finished, reward : {episode_reward}')
                self.save(save_path)
                print(f'epsilon : {self.epsilon}')
                self.update_epsilon()

        self.save(save_path)
        self.env.close()
        log_f.close()
env_name = 'PongNoFrameskip-v4'
# instantiate the environment, atari pong
#### log files for multiple runs are NOT overwritten
log_dir = "PPO_logs"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

#### get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)
env = gym.make(env_name,render_mode='human',difficulty=3)

# wrappe the environment
env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=True)
env = gym.wrappers.FrameStack(env, num_stack=4)
print(f'shape : {env.observation_space.shape}')
# instantiate the agent
agent = Agent(env, 10000, 256, 0.9999, 1.0, 0.996, 0.01, 0.0001)
print(f'Agent : {agent.network}')
agent.load('model.pth')
print(f'Lets play randomly before training  ! ')
agent.play()
# print a line of "
print("-"*50)
# play randomly before training
#agent.play()
print("-"*50)
agent.train_agent(1000000, 256, 100, 'model.pth')
