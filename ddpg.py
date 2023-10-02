import gym
# import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from Off_policy.DDPG import Agent

if __name__ == '__main__':
    config = {
        'env_name': 'Pendulum-v1',
        'gamma': 0.99,
        'tau': 0.005,
        'lr_critic': 0.0003,
        'lr_actor': 0.0003,
        'batch_size': 256,
        'buffer_size': 1000000,
        'seed': 1,
        'max_steps': 1000000,
        'tensorboard_path': 'runs/ddpg',
        'save_path': 'trained_models/ddpg/',
        'device': 'cpu'
    }


    env_name = config.get('env_name','CartPole-v1')
    gamma = config.get('gamma',0.99)
    tau = config.get('tau',0.005)
    lr_critic = config.get('lr_critic',0.0003)
    lr_actor = config.get('lr_actor',0.0003)
    batch_size = config.get('batch_size',256)
    buffer_size = config.get('buffer_size',1000000)
    seed = config.get('seed',1)
    max_steps = config.get('max_steps',1000000)
    device = config.get('device','cpu')
    tensorboard_path = config.get('tensorboard_path','runs/ddpg')
    save_path = config.get('save_path','trained_models/ddpg')

    save_path = save_path + env_name + '/'





    env = gym.make(env_name)


    writer = SummaryWriter(log_dir=tensorboard_path)


    agent = Agent(env=env,writer=writer,discount_factor=gamma,tau=tau,lr_critic=lr_critic,lr_actor=lr_actor,
                  replay_memory_size=buffer_size,replay_memory_sample_size=batch_size,device=device,
                  max_steps=max_steps,save_path=save_path)

    agent.rollout()



