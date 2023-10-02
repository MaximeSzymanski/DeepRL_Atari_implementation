import gym
from torch.utils.tensorboard import SummaryWriter
from Off_policy.DQN.agent import Agent


config = {
    'env_name': 'PongNoFrameskip-v4',
    'gamma': 0.99,
    'learning_rate': 0.0003,
    'seed': 1,
    'num_frames': 1e6,
    'batch_size': 32,
    'replay_buffer_size': 100000,
    'epsilon_decay': 0.99,
    'min_epsilon': 0.1,
    'train_freq': 4,
    'record_video' : False,
    'video_path' : 'videos/dqn',
    'device': 'cpu',
    'model_path': 'trained_models/dqn',
    'tensorboard_path': 'runs/dqn'
}

env_name = config.get('env_name','CartPole-v1')


env = gym.make(env_name)
#env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=True)
#env = gym.wrappers.FrameStack(env, num_stack=4)
# instantiate the agent
memory_size = config.get('replay_buffer_size',100000)
batch_size = config.get('batch_size',32)
gamma = config.get('gamma',0.99)
epsilon = config.get('epsilon',1.0)
epsilon_decay = config.get('epsilon_decay',0.99)
min_epsilon = config.get('min_epsilon',0.1)
train_freq = config.get('train_freq',4)
learning_rate = config.get('learning_rate',0.0003)
seed = config.get('seed',1)
sample_size = env.observation_space.shape[0]
device = config.get('device','cpu')
num_frames = config.get('num_frames',1e6)
model_path = config.get('model_path','trained_models/dqn') + env_name + '/'
writer = SummaryWriter(config.get('tensorboard_path','runs/dqn'))
env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=30, scale_obs=True)
env = gym.wrappers.FrameStack(env, num_stack=4)
agent = Agent(env=env,memory_size=memory_size, sample_size=sample_size, gamma=gamma, epsilon=epsilon,
             epsilon_decay=epsilon_decay,epsilon_min=min_epsilon, lr=learning_rate, seed=seed, device=device,batch_size=batch_size,
              model_path=model_path,writer=writer)
print(f'Agent : {agent.network}')

#agent.play()
# print a line of "

agent.train_agent(maximum_number_steps=num_frames, save_path= 'model.pth')
