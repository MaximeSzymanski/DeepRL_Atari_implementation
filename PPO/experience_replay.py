import numpy as np


class ExperienceReplay():
    def __init__(self,minibatch_size,buffer_size,state_size,num_workers =2 ,action_size=6,horizon=128):
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
        transformed_buffer_size =   (horizon,) + (self.num_worker,)
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

    def add_step(self,state,action,reward,next_state,done,value,olg_log_prob):
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
        indices = np.random.randint(0,self.size,size=self.minibatch_size)
        # return the minibatch
        return self.states[indices],self.actions[indices],self.rewards[indices],self.next_states[indices],self.dones[indices],self.olg_log_probs[indices],self.values[indices]

    def flatten_buffer(self):
        # flatten the buffer
        self.states = self.states.reshape(-1, 4, 84, 84)
        self.actions = self.actions.flatten()
        self.rewards = self.rewards.flatten()
        self.next_states = self.next_states.reshape(-1, 4, 84, 84)
        self.dones = self.dones.flatten()
        self.olg_log_probs = self.olg_log_probs.flatten()
        self.values = self.values.flatten()
        self.advantages = self.advantages.flatten()
    def clean_buffer(self):
        self.reset_buffer(self.horizon, self.state_size)

    def __len__(self):
        return self.size
