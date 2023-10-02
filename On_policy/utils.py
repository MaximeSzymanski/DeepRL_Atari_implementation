import numpy as np

from On_policy.PPO.agent import Agent
from On_policy.experience_replay import ExperienceReplay


def compute_advantages_ppo(experience_replay : ExperienceReplay, agent : Agent, gamma = 0.99, lamda = 0.95):

    for worker in range(experience_replay.num_worker):
        values = experience_replay.values[:,worker]

        advantages = np.zeros(agent.num_steps,dtype=np.float32)
        last_advantage = 0
        last_value = values[-1]
        for i in reversed(range(agent.num_steps)):
            mask = 1 - experience_replay.dones[i,worker]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = experience_replay.rewards[i,worker] + gamma * last_value - values[i]
            last_advantage = delta + gamma * lamda * last_advantage
            advantages[i] = last_advantage
            last_value = values[i]

        experience_replay.advantages[:,worker] = advantages
    pass
    experience_replay.flatten_buffer()
    advantages = experience_replay.advantages
    return advantages


def compute_advantages_a2c(agent):
        # simple advantage estimation, using TD(0) for the baseline
        for worker in range(agent.num_workers):
            values = agent.experience_replay.values[:, worker]

            advantages = np.zeros(agent.num_step, dtype=np.float32)

            for i in reversed(range(agent.num_step)):
                if i == agent.num_step - 1:
                    # if the state is terminal, then the next state value is 0
                    next_value = 0
                else:
                    next_value = values[i + 1]

                # TD error + bias
                advantages[i] = agent.experience_replay.rewards[i, worker] + agent.gamma * next_value * (
                        1 - agent.experience_replay.dones[i, worker]) - values[i]

            agent.experience_replay.advantages[:, worker] = advantages

        agent.experience_replay.flatten_buffer()
        advantages = agent.experience_replay.advantages
        return advantages
