import numpy as np

"""
This class implements replay buffer that will be used 
by DQN and DDPG.
"""
class ReplayBuffer():

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buff = []

    # adds tuple (s, a, r, s') to replay buffer
    def add(self, s1, a1, r1, s2):
        if len(self.buff) == self.buffer_size:
            self.buff = self.buff[1:]
        self.buff.append((s1, a1, r1, s2))

    # samples minibatch of size k from replay buffer
    def sample_minibatch(self, k):
        idx = np.random.choice(np.arange(self.buff), k, False)
        return self.buff[idx]

    

    
        
        
        
