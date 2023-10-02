import os

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Categorical

from On_policy.actor import Actor
from On_policy.cnn import CNN
from On_policy.critic import Critic
from On_policy.experience_replay import ExperienceReplay
from On_policy.utils import compute_advantages_a2c


class Agent(nn.Module):
    def __init__(self,action_size,lr,batch_size,num_step,
                 gamma,max_timesteps,num_workers,state_size,writer,
                 entropy_coef=0.01,
                 device='cpu',save_path='trained_models/a2c',
                 max_grad_norm=0.5):
        super(Agent,self).__init__()
        self.writer = writer
        self.save_path = save_path
        self.entropy_coef = entropy_coef
        self.device = device
        self.experience_replay = ExperienceReplay(minibatch_size=batch_size,buffer_size=num_step*num_workers,
                                         state_size=state_size,action_size=action_size,horizon=num_step)
        self.gamma = gamma
        self.lr = lr
        self.max_timesteps = max_timesteps
        self.num_workers = num_workers
        self.num_step = num_step
        self.batch_size = batch_size
        self.actor = Actor(action_size=action_size)
        self.critic = Critic(hidden_size=256)
        self.cnn = CNN()
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        self.number_epochs = 0

    def ortogonal_initialization(self):

        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = x / 255.0
        x = self.cnn(x)
        logits = self.actor(x)
        value = self.critic(x)
        dist = Categorical(logits)
        return dist, value

    def get_action(self, obs):
        with torch.no_grad():
            dist, value = self.forward(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach().numpy(), value.detach().numpy()

    def save_model(self):
        # create folder if not exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        path = self.save_path + 'model' + '.pth'
        torch.save(self.state_dict(), path)

    def rollout_episode(self,env):
        episode_index = 0
        time_step = 0

        state = env.reset()
        state = np.array(state)
        state = np.transpose(state, (0, 3, 1, 2))

        total_reward = np.zeros(env.num_envs)
        # reset the done flag
        total_time_step = 0
        step = 0
        step_counter = np.zeros(env.num_envs)
        print(f"num_envs : {env.num_envs}")
        max_episode = 10000
        for episode in range(max_episode):
            # reset the environment
            frac = 1.0 - (episode - 1.0) / max_episode
            lr_now = self.lr * frac
            self.optimizer.param_groups[0]['lr'] = lr_now
            # while the episode is not done
            for horizon in range(self.num_step):
                # increment the step counter
                step += 1
                time_step += self.num_workers
                total_time_step += 1
                # get the action

                action, log_prob, value = self.get_action(torch.from_numpy(state).to(self.device))
                # take the action

                next_state, reward, done_list, _ = env.step(action)
                next_state = np.array(next_state)
                next_state = np.transpose(next_state, (0, 3, 1, 2))
                # add the step to the buffer
                done_to_add = [1 if done else 0 for done in done_list]
                self.experience_replay.add_step(state, action, reward, next_state, done_to_add, value, log_prob)
                # update the total reward
                total_reward += reward

                # add 1 to each value of the numpy array
                step_counter += 1

                for worker in range(env.num_envs):
                    if done_list[worker] == True:
                        self.writer.add_scalar('Reward', total_reward[worker], total_time_step+worker)
                        print(
                            f'Episode  finished after {step_counter[worker]} steps, total reward : {total_reward[worker]},total time step : {total_time_step}')
                        total_reward[worker] = 0
                        episode_index += 1
                        step_counter[worker] = 0
                        done_list[worker] = False
                # update the state
                state = next_state

            print(f"-" * 50)
            print(f"updating the agent...")
            print(f"-" * 50)
            self.train_agent()

            self.save_model()


    def train_agent(self):


        advantages = compute_advantages_a2c(self)
        # convert the data to torch tensors
        states = torch.from_numpy(self.experience_replay.states).to(self.device)
        actions = torch.from_numpy(self.experience_replay.actions).to(self.device)
        advantages = torch.from_numpy(advantages).to(self.device)
        values = torch.from_numpy(self.experience_replay.values).to(self.device)

        returns = advantages + values


        # split the data into batches
        numer_of_samples = self.num_step * self.num_workers

        number_mini_batch =  numer_of_samples // self.experience_replay.minibatch_size
        assert number_mini_batch > 0 , "batch size is too small"
        assert numer_of_samples % self.experience_replay.minibatch_size  == 0 , "batch size is not a multiple of the number of samples"

        indices = np.arange(numer_of_samples)
        np.random.shuffle(indices)

        for batch_index in range(number_mini_batch):

                start = batch_index * self.experience_replay.minibatch_size
                end = (batch_index + 1) * self.experience_replay.minibatch_size
                indice_batch = indices[start:end]
                advantages_batch = advantages[indice_batch]
                normalized_advantages = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                new_dist, new_values = self.forward(states[indice_batch])
                new_log_probs = new_dist.log_prob(actions[indice_batch])
                entropy = new_dist.entropy().mean()
                actor_loss = -(normalized_advantages * new_log_probs).mean()
                critic_loss = (returns[indice_batch] - new_values).pow(2).mean()


                self.writer.add_scalar('entropy', entropy, self.number_epochs)
                loss = actor_loss + critic_loss - self.entropy_coef * entropy
                self.writer.add_scalar('Value loss ', critic_loss, self.number_epochs)
                self.writer.add_scalar('Policy loss ', actor_loss, self.number_epochs)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.number_epochs += 1

        self.experience_replay.clean_buffer()

    def start_training(self,env):

        self.rollout_episode(env)
