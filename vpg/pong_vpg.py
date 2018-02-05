import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

STATE_DIM = 80 * 80
HIDDEN1 = 200

class NNEstimator(nn.Module):

    def __init__(self):
        super(NNEstimator, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.softmax(x)
        x = torch.log(x)
        return x


class VanillaPolicyGradientAgent:

    def __init__(self):
        # Construction of NN
        self.nn = NNEstimator()
        pass

    def update(self):
        lr = 0.01
        for f in self.nn.parameters():
            f.data.add_(f.grad.data * lr)

    # Given state, function returns tuple (action, grad_log_probability) for that action
    def get_logprobs(self, state):
        state = np.expand_dims(state, axis=0)

        tstate = torch.from_numpy(state)
        tstate = tstate.type(torch.FloatTensor)

        probs = self.nn.forward(Variable(tstate))

        return probs

def downsample(x):
    x = x[35:195]
    x = x[::2, ::2, 0]
    x[x == 109] = 0
    x[x == 144] = 0
    x[x != 0] = 1
    return x

def get_real_action(action):
    if action == 1:
        return 2
    elif action == 2:
        return 3
    else:
        return 0

def discount_rewards(r):
    dr = []
    curr = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            curr = 0
        curr = curr * 0.99 + r[t]
        dr.append(curr)
    dr.reverse()
    return dr
    
def main():
    env = gym.make('Pong-v0')
    vpg = VanillaPolicyGradientAgent()

    for it in range(3000):
        observation = env.reset()
        observation = downsample(observation)
        prev_observation = np.zeros_like(observation)
        
        rewards = []
        all_logprobs = []

        while True:
            vpg.nn.zero_grad()
            # if it % 10 == 0:
            #     env.render()

            #action = env.action_space.sample()
            logprobs = vpg.get_logprobs(observation - prev_observation)
            all_logprobs.append(logprobs)

            topv, topi = torch.topk(logprobs, 1, dim=1)
            action = topi.data[0][0]

            
            prev_observation = observation
            observation, reward, done, info = env.step(get_real_action(action))
            observation = downsample(observation)

            rewards.append(reward)
            
            if done:
                break

        rewards = discount_rewards(rewards)

        print(it+1, ' total reward...', np.sum(rewards))

        g = 0

        for i, logprobs in enumerate(all_logprobs):
            topv, topi = torch.topk(logprobs, 1, dim=1)
            action = topi.data[0][0]

            rews = torch.zeros(logprobs.size())
            rews[0][action] = rewards[i]
            rews = Variable(rews)

            g += torch.dot(logprobs, rews)
            
        g.backward()


if __name__ == '__main__':
    main()
