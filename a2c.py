from On_policy.A2C.agent import Agent
import torch
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3



config = {
    'env_name': 'PongNoFrameskip-v4',
    'gamma': 0.99,
    'num_workers': 8,
    'num_step': 128,
    'lr': 0.0001,
    'entropy_coef': 0.01,
    'max_timesteps': 10000000,
    'batch_size': 256,
    'save_path': 'trained_models/a2c',
    'device': 'cpu',
    'tensorboard_log': 'tensorboard_logs/a2c',
    'max_grad_norm': 0.5
}

env_name = config.get('env_name', 'CartPole-v1')

log_dir = config.get('tensorboard_path', 'runs/a2c') + env_name
writer = SummaryWriter(log_dir=log_dir)
device = torch.device(config.get('device', 'cuda:0'))
gamma = config.get('gamma', 0.99)
max_timesteps = config.get('max_timesteps', 1000000)
lr = config.get('lr', 0.0003)
num_workers = config.get('num_envs', 1)
seed = config.get('seed', 1)
batch_size = config.get('batch_size', 64)
num_steps = config.get('num_steps', 2048)
entropy_coef = config.get('entropy_coef', 0.01)
env = stable_baselines3.common.env_util.make_atari_env(env_name, n_envs=num_workers, seed=seed)
env = stable_baselines3.common.vec_env.vec_frame_stack.VecFrameStack(env, n_stack=4)
# env = stable_baselines3.common.vec_env.VecVideoRecorder(env, video_folder=config.get('video_path','videos/ppo') + env_name, record_video_trigger=lambda x: x == 0, name_prefix="ppo"+env_name)
state_size = (4, 84, 84)
action_size = env.action_space.n
save_path = config.get('save_path', 'trained_models/a2c') + env_name + '/'


agent = Agent(action_size=action_size,
              state_size=state_size,
              lr=lr,
              batch_size=batch_size,
              num_step=num_steps,
              gamma=gamma,
              max_timesteps=max_timesteps,
              num_workers=num_workers,
              writer=writer,
              entropy_coef=entropy_coef,
              device=device,
              save_path=save_path,
              max_grad_norm=config.get('max_grad_norm', 0.5))
agent.to(device)

agent.start_training(env)


