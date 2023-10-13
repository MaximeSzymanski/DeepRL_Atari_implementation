import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
# import writer tensorboard
import stable_baselines3
from torch.utils.tensorboard import SummaryWriter
import unittest
import torch.nn.functional as F
import gym
import os

if __name__ == '__main__':
    log_running_reward = 0
    log_running_episode = 0
    writer = SummaryWriter(log_dir='breakout/ppo')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    env_name = 'CartPole-v1'
    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    lr = 0.0025


    class ExperienceReplay():
        def __init__(self, minibatch_size, buffer_size, state_size, num_workers=2, action_size=6, horizon=128):
            self.minibatch_size = minibatch_size
            self.buffer_size = buffer_size
            self.state_size = state_size
            print(f'buffer size : {buffer_size}')
            print(f'horizon : {horizon}')
            print(f'number of workers : {num_workers}')
            self.num_worker = num_workers
            self.horizon = horizon
            self.reset_buffer(horizon, state_size)
            print(f'buffer size : {self.states.shape}')

        def reset_buffer(self, horizon, state_size):
            transformed_buffer_size = (horizon,) + (self.num_worker,)
            buffer_state_size = transformed_buffer_size + state_size

            self.actions = np.empty(transformed_buffer_size, dtype=np.int32)
            self.rewards = np.empty(transformed_buffer_size, dtype=np.float32)
            self.states = np.empty(buffer_state_size, dtype=np.float32)
            self.next_states = np.empty(buffer_state_size, dtype=np.float32)
            self.dones = np.empty(transformed_buffer_size, dtype=np.int32)
            self.olg_log_probs = np.empty(transformed_buffer_size, dtype=np.float32)
            self.advantages = np.empty(transformed_buffer_size, dtype=np.float32)
            self.values = np.empty(transformed_buffer_size, dtype=np.float32)

            self.head = 0
            self.size = 0

        def add_step(self, state, action, reward, next_state, done, value, olg_log_prob):
            # assert the buffer is not full
            assert self.size < self.buffer_size, "Buffer is full"

            self.states[self.head] = state
            self.actions[self.head] = action
            value = np.squeeze(value)
            self.values[self.head] = value
            self.olg_log_probs[self.head] = olg_log_prob
            self.rewards[self.head] = reward
            self.next_states[self.head] = next_state
            self.dones[self.head] = done
            self.head = (self.head + 1) % self.horizon
            self.size += 1
            # check if the buffer is full

        def get_minibatch(self):
            # assert the buffer is not empty
            assert self.size > self.minibatch_size, "Buffer is empty"
            # get random indices
            indices = np.random.randint(0, self.size, size=self.minibatch_size)
            # return the minibatch
            return self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices], \
            self.dones[indices], self.olg_log_probs[indices], self.values[indices]

        def flatten_buffer(self):
            # flatten the buffer
            self.states = self.states.reshape(-1,4)
            self.actions = self.actions.flatten()
            self.rewards = self.rewards.flatten()
            self.next_states = self.next_states.reshape(-1, 4)
            self.dones = self.dones.flatten()
            self.olg_log_probs = self.olg_log_probs.flatten()
            self.values = self.values.flatten()
            self.advantages = self.advantages.flatten()

        def clean_buffer(self):
            self.reset_buffer(self.horizon, self.state_size)

        def __len__(self):
            return self.size


    class Agent(nn.Module):

        def __init__(self, state_size, action_size, num_workers=8, num_steps=128, batch_size=256):
            super(Agent, self).__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 9 * 9, 256),

            )
            self.actor = nn.Sequential(
                nn.Linear(4, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
                nn.Softmax(dim=-1)

            )
            self.critic = nn.Sequential(
                nn.Linear(4, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            self.number_epochs = 0
            print(self.actor)
            print(self.critic)
            self.num_workers = num_workers
            self.num_steps = num_steps
            self.batch_size = batch_size
            self.ortogonal_initialization()
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
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.critic.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            for m in self.cnn.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            # = x / 255.0
            #x = self.cnn(x)
            logits = self.actor(x)
            value = self.critic(x)
            dist = Categorical(logits)

            return dist, value

        def get_action(self, obs, deterministic=False):
            with torch.no_grad():
                dist, value = self.forward(obs)
                if deterministic:
                    action = torch.argmax(dist.probs).unsqueeze(0)

                else:
                    action = dist.sample()

                log_prob = dist.log_prob(action)

            return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), value.cpu().detach().numpy()

        def decay_learning_rate(self, optimizer, decay_rate=0.99):
            print("Decaying learning rate")
            writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], self.number_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate

        def save_model(self, path='ppo.pth'):
            torch.save(self.state_dict(), path)

        def load_model(self, path='ppo.pth'):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def compute_advantages(experience_replay: ExperienceReplay, agent: Agent, gamma=0.99, lamda=0.95):

        for worker in range(experience_replay.num_worker):
            if experience_replay.dones[-1, worker] == 1:
                R = 0
            else:
                R = experience_replay.values[-1, worker]

            for step in reversed(range(experience_replay.horizon)):

                R = experience_replay.rewards[step, worker] + gamma * R * (1 - experience_replay.dones[step, worker])
                experience_replay.advantages[step, worker] = R - experience_replay.values[step, worker]


        experience_replay.flatten_buffer()
        advantages = experience_replay.advantages
        return advantages

    def test_agent(env, agent, render=False):
        episode_index = 0
        time_step = 0


        state , _  = env.reset()
        #state = np.array(state)
        #state = np.transpose(state, (0, 3, 1, 2))
        # reset the buffer

        # reset the total reward
        total_reward = np.zeros(env.num_envs)
        # reset the done flag
        log_freq = 100
        total_time_step = 0
        done = False
        step = 0
        step_counter = np.zeros(env.num_envs)
        print(f"num_envs : {env.num_envs}")
        for episode in range(1000000):
            # reset the environment

            # while the episode is not done
            for horizon in range(agent.num_steps):
                env.render()
                # increment the step counter
                step += 1
                time_step += agent.num_workers
                total_time_step += 1
                # get the action

                action, log_prob, value = agent.get_action(torch.from_numpy(state).to(device),deterministic=True)
                # take the action

                next_state, reward, done_list, _ , _ = env.step(action)
                next_state = np.array(next_state)
                #next_state = np.transpose(next_state, (0, 3, 1, 2))
                # add the step to the buffer
                done_to_add = [1 if done else 0 for done in done_list]
                # update the total reward
                total_reward += reward

                # add 1 to each value of the numpy array
                step_counter += 1

                for worker in range(env.num_envs):
                    if done_list[worker] == True:
                        writer.add_scalar('Reward', total_reward[worker], total_time_step)
                        print(
                            f'Episode  finished after {step_counter[worker]} steps, total reward : {total_reward[worker]},total time step : {time_step}')
                        total_reward[worker] = 0
                        episode_index += 1
                        step_counter[worker] = 0
                        done_list[worker] = False

                # update the state
                state = next_state
                # render the environment
                # log in logging file



    def rollout_episode(env, agent, experience_replay, render=False):
        episode_index = 0
        time_step = 0
        log_running_reward = 0
        log_running_episodes = 0
        state  = env.reset()
        state = np.array(state)
        #state = np.transpose(state, (0, 3, 1, 2))
        print(f'state shape : {state.shape}')
        # reset the buffer

        # reset the total reward
        total_reward = np.zeros(env.num_envs)
        # reset the done flag
        log_freq = 100
        total_time_step = 0
        done = False
        step = 0
        step_counter = np.zeros(env.num_envs)
        print(f"num_envs : {env.num_envs}")
        for episode in range(1000000):
            # reset the environment

            # while the episode is not done
            for horizon in range(agent.num_steps):
                env.render()
                # increment the step counter
                step += 1
                time_step += agent.num_workers
                total_time_step += 1
                # get the action

                action, log_prob, value = agent.get_action(torch.from_numpy(state).to(device))
                # take the action

                next_state, reward, done_list, _ = env.step(action)
                next_state = np.array(next_state)
                #next_state = np.transpose(next_state, (0, 3, 1, 2))
                # add the step to the buffer
                done_to_add = [1 if done else 0 for done in done_list]

                experience_replay.add_step(state, action, reward, next_state, done_to_add, value, log_prob)
                # update the total reward
                total_reward += reward

                # add 1 to each value of the numpy array
                step_counter += 1

                for worker in range(env.num_envs):
                    if done_list[worker] == True:
                        writer.add_scalar('Reward', total_reward[worker], total_time_step)
                        print(
                            f'Episode  finished after {step_counter[worker]} steps, total reward : {total_reward[worker]},total time step : {time_step}')
                        total_reward[worker] = 0
                        episode_index += 1
                        step_counter[worker] = 0
                        done_list[worker] = False

                # update the state
                state = next_state
                # render the environment
                # log in logging file

            print(f"-" * 50)
            print(f"updating the agent...")
            print(f"-" * 50)
            train_agent(agent, experience_replay)
            #if episode % 100 == 0:
                #Agent.save_model(f'breakout/ppo_{episode}.pth')

            # return the total


    def train_agent(agent: Agent, experience_replay: ExperienceReplay):

        advantages = compute_advantages(experience_replay, agent, gamma=0.99, lamda=0.95)
        # convert the data to torch tensors
        states = torch.from_numpy(experience_replay.states).to(device)
        actions = torch.from_numpy(experience_replay.actions).to(device)
        old_log_probs = torch.from_numpy(experience_replay.olg_log_probs).to(device).detach()

        advantages = torch.from_numpy(advantages).to(device)
        values = torch.from_numpy(experience_replay.values).to(device)

        returns = advantages + values

        # split the data into batches
        numer_of_samples = agent.num_steps * experience_replay.num_worker

        number_mini_batch = numer_of_samples // experience_replay.minibatch_size
        assert number_mini_batch > 0, "batch size is too small"
        assert numer_of_samples % experience_replay.minibatch_size == 0, "batch size is not a multiple of the number of samples"

        indices = np.arange(numer_of_samples)
        np.random.shuffle(indices)
        for batch_index in range(number_mini_batch):
                start = batch_index * experience_replay.minibatch_size
                end = (batch_index + 1) * experience_replay.minibatch_size
                indice_batch = indices[start:end]
                new_dist, new_values = agent(states[indice_batch])

                log_pi = new_dist.log_prob(actions[indice_batch])

                actor_loss = -(log_pi * advantages[indice_batch]).mean()

                critic_loss = (returns[indice_batch] - new_values).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        experience_replay.clean_buffer()
        # agent.decay_learning_rate(optimizer)

        # create the dataset


    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    num_workers = 8
    num_steps = 5
    batch_size = 40

    env = stable_baselines3.common.vec_env.DummyVecEnv([lambda: env for i in range(num_workers)])

    state_size = (4,)

    print(f'state size : {state_size}')
    action_size = 2
    print(f'action size : {action_size}')
    ExperienceReplay = ExperienceReplay(batch_size, num_steps * num_workers, state_size=state_size,
                                        num_workers=num_workers, action_size=1, horizon=num_steps)

    Agent = Agent(state_size, action_size, num_workers=num_workers, num_steps=num_steps, batch_size=batch_size)
    Agent.to(device)
    optimizer = torch.optim.Adam(Agent.parameters(), lr=lr)
    number_of_episodes = 0

    rollout_episode(env,agent=Agent,experience_replay=ExperienceReplay)