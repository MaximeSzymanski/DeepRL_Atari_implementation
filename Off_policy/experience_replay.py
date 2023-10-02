import numpy as np


class Experience_replay():

    # state is a numpy array of shape (memory_size, state_size)


    memory_size = 0
    sample_size = 0

    def __init__(self,memory_size,sample_size,state_size):
        self.memory_size = memory_size
        self.sample_size = sample_size
        shape_state = (memory_size,) + state_size
        shape = (memory_size,1)
        self.state = np.empty(shape_state)
        print(f'state shape: {self.state.shape}')
        self.action = np.empty(shape)
        self.reward = np.empty(shape)
        self.next_state = np.empty(shape_state)
        self.done = np.empty(shape)
        self.head = 0
        self.size = 0


    def add(self,state,action,reward,next_state,done):
        self.state[self.head] = state
        self.action[self.head] = action
        self.reward[self.head] = reward
        self.next_state[self.head] = next_state
        self.done[self.head] = done
        self.head = (self.head + 1) % self.memory_size
        if self.size < self.memory_size:
            self.size += 1


    def sample(self,batch_index):

        return self.state[batch_index],self.action[batch_index],self.reward[batch_index],self.next_state[batch_index],self.done[batch_index]
