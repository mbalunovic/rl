import gym
import numpy as np
from rl.lib.policy_gradient.ddpg import DDPGAgent
from rl.policies.neural_net_policies import NNContinuousPolicy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

NUM_EPISODES = 10000000000
MAX_ITERATIONS = 100000000
REFIT_IT = 1000
K = 1000000
REPEAT = 5
MEAN = 1000

class NNCritic(nn.Module):

  def __init__(self, state_dim, action_dim):
    super(NNCritic, self).__init__()
    self.input_dim = state_dim + action_dim
    
    self.fc1 = nn.Linear(self.input_dim, 50)
    self.fc2 = nn.Linear(50, 10)
    self.fc3 = nn.Linear(10, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  def q(self, s, a):
    s = Variable(torch.from_numpy(s).type(torch.FloatTensor))
    x = torch.cat((s, a), 1)
    return self.forward(x)

def main():
  env = gym.make('Pendulum-v0')

  actor = NNContinuousPolicy(3, 1, [30, 30], bound=True)
  critic = NNCritic(3, 1)
  ddpg_agent = DDPGAgent(3, 1, 0.99, 0.0001, actor, critic)

  curr_rewards = []
  
  for episode in range(1, NUM_EPISODES):
    print(ddpg_agent.train_episode(env=env, tot_iterations=1000, render=False, action_transformer=lambda a: 2 * a))
    

if __name__ == '__main__':
    main()
