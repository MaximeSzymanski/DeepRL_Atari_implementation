import numpy as np
import torch
from torch.nn import functional as F

from PPO.agent import Agent
from PPO.experience_replay import ExperienceReplay
from PPO.utils import compute_advantages




def test_agent(env,agent):
    state = env.reset()
    print(state.shape)
    state = np.array(state)
    state = np.transpose(state, (0, 3, 1, 2))
    total_reward = 0
    done = False

    while True:
        env.render(mode='human')
        action, log_prob, value = agent.get_action(torch.from_numpy(state).float().to(agent.device))
        next_state, reward, done, _ = env.step(action)
        next_state = np.array(next_state)
        next_state = np.transpose(next_state, (0, 3, 1, 2))

        state = next_state
        total_reward += reward
    print(f"total reward : {total_reward}")
    env.close()
    return total_reward


def rollout_episode(env, agent, experience_replay, render=False,writer=None,config=None):
    episode_index = 0
    time_step = 0

    state= env.reset()
    state = np.array(state)
    state = np.transpose(state, (0, 3, 1, 2))
    print(f'state shape : {state.shape}')
    # reset the buffer

    # reset the total reward
    total_reward = np.zeros(env.num_envs)
    # reset the done flag
    total_time_step = 0
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
                    print(f'Episode  finished after {step_counter[worker]} steps, total reward : {total_reward[worker]},total time step : {total_time_step}')
                    total_reward[worker] = 0
                    episode_index += 1
                    step_counter[worker] = 0
                    done_list[worker] = False
            # update the state
            state = next_state


        print(f"-"*50)
        print(f"updating the agent...")
        print(f"-"*50)

        train_agent(agent, experience_replay)



        # return the total


def train_agent(agent : Agent, experience_replay : ExperienceReplay):


    advantages = compute_advantages(experience_replay, agent, gamma=0.99, lamda=0.95)
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
    for _ in range(agent.K_epochs):
        for batch_index in range(number_mini_batch):

            start = batch_index * experience_replay.minibatch_size
            end = (batch_index + 1) * experience_replay.minibatch_size
            indice_batch = indices[start:end]
            advantages_batch = advantages[indice_batch]
            normalized_advantages = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            agent.number_epochs += 1
            new_dist, new_values = agent(states[indice_batch])
            log_pi = new_dist.log_prob(actions[indice_batch])

            ratio = torch.exp(log_pi - old_log_probs[indice_batch].detach())
            surr1 = ratio * normalized_advantages
            surr2 = torch.clamp(ratio, 1 - agent.clip_param, 1 + agent.clip_param) * normalized_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(), returns[indice_batch])


            entropy_loss = new_dist.entropy().mean()
            agent.writer.add_scalar('entropy', entropy_loss, agent.number_epochs)
            loss = actor_loss + agent.value_loss_coef * critic_loss - agent.entropy_coef * entropy_loss

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
    experience_replay.clean_buffer()
    agent.decay_learning_rate()
