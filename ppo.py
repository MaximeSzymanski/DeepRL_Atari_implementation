import torch
# import writer tensorboard
import stable_baselines3
from torch.utils.tensorboard import SummaryWriter
import gym
import os

from PPO.agent import Agent
from PPO.experience_replay import ExperienceReplay
from PPO.interaction_env import test_agent, rollout_episode

if __name__ == '__main__':
    config = {
        'env_name': 'BreakoutNoFrameskip-v4',
        'gamma': 0.99,
        'gae': 0.95,
        'clip': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'num_steps': 128,
        'num_epochs': 10,
        'batch_size': 256,
        'lr': 0.0003,
        'eps': 1e-5,
        'max_grad_norm': 0.5,
        'seed': 1,
        'num_workers': 8,
        'num_frames': 1e6,
        'tensorboard_path': 'runs/ppo',
        'save_path': 'trained_models/ppo/',
        'record_video' : False,
        'video_path' : 'videos/ppo',
        'device': 'cpu',
        'learning_rate_decay': 0.99
    }
    env_name = config.get('env_name','CartPole-v1')

    log_dir = config.get('tensorboard_path','runs/ppo') + env_name
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device(config.get('device','cuda:0'))

    lr = config.get('lr',0.0003)
    num_workers = config.get('num_envs',1)
    seed = config.get('seed',1)
    batch_size = config.get('batch_size',64)
    num_steps = config.get('num_steps',2048)
    value_loss_coef = config.get('value_loss_coef',0.5)
    entropy_coef = config.get('entropy_coef',0.01)
    clip_grad_norm = config.get('clip_grad_norm',0.5)
    num_epochs = config.get('num_epochs',10)
    clip_param = config.get('clip',0.2)
    env = stable_baselines3.common.env_util.make_atari_env(env_name, n_envs=num_workers, seed=seed)
    env = stable_baselines3.common.vec_env.vec_frame_stack.VecFrameStack(env, n_stack=4)
    max_grad_norm = config.get('max_grad_norm',0.5)
    #env = stable_baselines3.common.vec_env.VecVideoRecorder(env, video_folder=config.get('video_path','videos/ppo') + env_name, record_video_trigger=lambda x: x == 0, name_prefix="ppo"+env_name)
    state_size = (4,84,84)
    action_size = env.action_space.n
    save_path = config.get('save_path','trained_models/ppo') + env_name + '/'
    ExperienceReplay = ExperienceReplay(num_steps, num_steps * num_workers, state_size=state_size, num_workers=num_workers, action_size=action_size, horizon=num_steps)
    Agent = Agent(state_size, action_size, num_workers=num_workers, num_steps=num_steps, batch_size=batch_size,device=device,lr=lr,writer=writer,learining_rate_decay = config.get('learning_rate_decay',0.99),
                 value_loss_coef=value_loss_coef, entropy_coef=entropy_coef, clip_grad_norm=clip_grad_norm, K_epochs=num_epochs,clip_param=clip_param,
                  save_path=save_path)

    Agent.to(device)
    #Agent.load_state_dict(torch.load('ppo_4500.pth',map_location=device))
    number_of_episodes = 0
    while(True):
        rollout_episode(env,Agent,ExperienceReplay)


