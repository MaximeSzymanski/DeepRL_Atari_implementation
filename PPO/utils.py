import numpy as np

from PPO.agent import Agent
from PPO.experience_replay import ExperienceReplay


def compute_advantages(experience_replay : ExperienceReplay, agent : Agent, gamma = 0.99, lamda = 0.95):

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
