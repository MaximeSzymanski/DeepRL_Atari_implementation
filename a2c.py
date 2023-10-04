import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3
import numpy as np
import torch.nn.functional as F
import gym
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
        self.states = self.states.reshape(-1, 4)
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
    def __init__(self,action_size,state_size,lr,
            gamma,buffer_size,batch_size,num_step,writer,
                 device,num_worker):
        super(Agent,self).__init__()
        self.action_size = action_size
        self.state_size = state_size
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_steps = num_step
        self.writer = writer
        self.device = device
        self.total_step = 0
        self.total_update = 0
        self.num_worker = num_worker

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),

        )
        print(f'action size : {action_size}')
        self.actor = nn.Sequential(
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)

        )
        self.critic = nn.Sequential(
            nn.Linear(256, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)




def compute_advantages(experience_replay : ExperienceReplay, agent : Agent,gamma = 0.99, lamda = 0.95):

        for worker in range(experience_replay.num_worker):
            values = experience_replay.values[:,worker]

            advantages = np.zeros(agent.num_steps,dtype=np.float32)
            last_advantage = 0
            next_value = 0
            for i in reversed(range(agent.num_steps)):
                mask = 1 - experience_replay.dones[i,worker]
                if i == agent.num_steps - 1:
                    next_value = 0
                else:
                    next_value = values[i+1]

                advantages[i]  = experience_replay.rewards[i] + mask * gamma * next_value - values[i]

            experience_replay.advantages[:,worker] = advantages
        pass
        experience_replay.flatten_buffer()
        advantages = experience_replay.advantages
        return advantages
def train_agent(agent : Agent, experience_replay : ExperienceReplay):

        advantages = compute_advantages(experience_replay,agent,gamma=0.99,lamda=0.95)
        # convert the data to torch tensors
        states = torch.from_numpy(experience_replay.states).to(agent.device)
        actions = torch.from_numpy(experience_replay.actions).to(agent.device)
        old_log_probs = torch.from_numpy(experience_replay.olg_log_probs).to(agent.device).detach()

        advantages = torch.from_numpy(advantages).to(agent.device)
        values = torch.from_numpy(experience_replay.values).to(agent.device)

        returns = advantages + values


        # split the data into batches
        numer_of_samples = agent.num_steps * experience_replay.num_worker

        number_mini_batch =  numer_of_samples // experience_replay.minibatch_size
        assert number_mini_batch > 0 , "batch size is too small"
        assert numer_of_samples % experience_replay.minibatch_size  == 0 , "batch size is not a multiple of the number of samples"

        indices = np.arange(numer_of_samples)
        np.random.shuffle(indices)
        for batch_index in range(number_mini_batch):

                start = batch_index * experience_replay.minibatch_size
                end = (batch_index + 1) * experience_replay.minibatch_size
                indice_batch = indices[start:end]
                advantages_batch = advantages[indice_batch]
                normalized_advantages = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                agent.number_epochs += 1

                new_dist, new_values = agent(states[indice_batch])
                new_log_probs = new_dist.log_prob(actions[indice_batch])

                actor_loss = -new_log_probs * normalized_advantages.detach()

                critic_loss = F.mse_loss(new_values, returns[indice_batch].detach())


                entropy_loss = new_dist.entropy().mean()
                agent.writer.add_scalar('entropy', entropy_loss, agent.number_epochs)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

        experience_replay.clean_buffer()

def rollout_episode(env, agent, experience_replay, render=False):
        episode_index = 0
        time_step = 0
        log_running_reward = 0
        log_running_episodes = 0
        state= env.reset()
        state = np.array(state)
        state = np.transpose(state, (0, 3, 1, 2))
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
        for episode in range(10000):
            # reset the environment


            # while the episode is not done
            for horizon in range(agent.num_steps):
                # increment the step counter
                step += 1
                time_step += agent.num_workers
                total_time_step += 1
                # get the action

                action, log_prob,value  = agent.get_action(torch.from_numpy(state).to(agent.device))
                # take the action

                next_state, reward, done_list, _ = env.step(action)
                next_state = np.array(next_state)
                next_state = np.transpose(next_state, (0, 3, 1, 2))
                # add the step to the buffer
                done_to_add = [1 if done else 0 for done in done_list]
                experience_replay.add_step(state, action, reward, next_state, done_to_add,  value, log_prob)
                # update the total reward
                total_reward += reward

                # add 1 to each value of the numpy array
                step_counter += 1


                for worker in range(env.num_envs):
                    if done_list[worker] == True:

                        agent.writer.add_scalar('Reward', total_reward[worker], total_time_step)
                        print(f'Episode  finished after {step_counter[worker]} steps, total reward : {total_reward[worker]},total time step : {time_step}')
                        total_reward[worker] = 0
                        episode_index += 1
                        step_counter[worker] = 0
                        done_list[worker] = False

                # update the state
                state = next_state
                # render the environment
                # log in logging file

            print(f"-"*50)
            print(f"updating the agent...")
            print(f"-"*50)
            train_agent(agent, experience_replay)
            if episode % 100 ==0:
                Agent.save_model(f'breakout/ppo_{episode}.pth')


if __name__ == '__main__':
    writer = SummaryWriter(log_dir='breakout/ppo')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    lr = 0.00025

    env_name = "PongNoFrameskip-v4"
    env = gym.make(env_name,render_mode="rgb_array")
    num_workers = 8
    num_steps = 128
    batch_size = 512


    env = stable_baselines3.common.env_util.make_atari_env(env_name, n_envs=num_workers, seed=0)
    env = stable_baselines3.common.vec_env.vec_frame_stack.VecFrameStack(env, n_stack=4)
    env = stable_baselines3.common.vec_env.VecVideoRecorder(venv=env,video_folder ='breakout/video',record_video_trigger=lambda x : not x % 100000)

    state_size = (4,84,84)
    gamma = 0.99
    print(f'state size : {state_size}')
    action_size = env.action_space.n
    print(f'action size : {action_size}')
    ExperienceReplay = ExperienceReplay(batch_size,num_steps*num_workers,state_size=state_size,num_workers=num_workers,action_size=1,horizon=num_steps)


    Agent = Agent(state_size=state_size,action_size=action_size,lr=lr,gamma=gamma,
                  buffer_size=num_workers*num_steps,writer=writer,device=device,num_worker=num_workers,num_step=num_steps,batch_size=batch_size)
    Agent.to(device)
    optimizer = torch.optim.Adam(Agent.parameters(),lr=lr)
    number_of_episodes = 0

    for i in range(100000):
        rollout_episode(env,Agent,ExperienceReplay,render=True)
        number_of_episodes+= 1