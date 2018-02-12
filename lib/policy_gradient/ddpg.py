import copy
import numpy as np
import torch

from rl.utils.experience_replay import ReplayBuffer
from rl.exploration.noise import OUProcess
from torch.autograd import Variable
from torch.nn import MSELoss
from torch.optim import Adam

REPLAY_BUFFER_SIZE = 10000
MINIBATCH_SIZE = 250
TAU = 0.999
WINDOW = 40

class DDPGAgent:
    
    def __init__(self, state_dim, action_dim, discount_rate, learning_rate, actor, critic):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.exp_replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.ou = OUProcess(sigma=0.5)

        self.actor = actor
        self.critic = critic
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.target_actor.requires_grad = False
        self.target_critic.requires_grad = False

        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)

        self.criterion = MSELoss()
        self.noise_p = 1
        self.noise_decay = 0.999

        self.running_rewards = []

    def move_target(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data = tau * target_param.data + (1 - tau) * source_param.data
        
    def train_episode(self, env, tot_iterations=10**10, render=False, action_transformer=None):
        ret_reward = 0
        ret_iterations = tot_iterations
        
        obs_new = env.reset()
        if render:
            env.render()
            
        for t in range(tot_iterations):
            self.noise_p *= self.noise_decay
            obs = obs_new
            
            #a = (1 - self.noise_p) * self.actor.get_action(obs) + self.noise_p * self.ou.sample()
            a = self.actor.get_action(np.array([obs])) * self.ou.sample()
            
            if action_transformer is not None:
                a = action_transformer(a)

            obs_new, reward, done, info = env.step(a.data.numpy())
            if type(reward) != float:
                reward = reward.item()
            
            self.exp_replay.add(obs, a, reward, obs_new)
            ret_reward += reward

            if done:
                ret_iterations = t
                break

            if self.exp_replay.size() < MINIBATCH_SIZE:
                continue

            mini_batch = self.exp_replay.sample_minibatch(MINIBATCH_SIZE)
            batch_states = np.zeros((MINIBATCH_SIZE, self.state_dim))
            batch_actions = np.zeros((MINIBATCH_SIZE, self.action_dim))
            batch_rewards = np.zeros((MINIBATCH_SIZE, 1))
            batch_new_states = np.zeros((MINIBATCH_SIZE, self.state_dim))
            for i, (s,a,r,s_new) in enumerate(mini_batch):
                batch_states[i] = s.flatten()
                batch_rewards[i,0] = r
                batch_new_states[i] = s_new.flatten()
                batch_actions[i] = a.data.numpy().flatten()
            batch_actions = Variable(torch.from_numpy(batch_actions).type(torch.FloatTensor))
            batch_rewards = Variable(torch.from_numpy(batch_rewards).type(torch.FloatTensor))

            self.actor.zero_grad()
            actor_loss = 0
            batch_actor_actions = self.actor.get_action(batch_states)
            actor_loss = -self.critic.q(batch_states, batch_actor_actions)
            actor_loss = actor_loss.mean()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            batch_target_actions = self.target_actor.get_action(batch_states)

            self.critic.zero_grad()
            critic_loss = 0
            target_q = batch_rewards + self.discount_rate * self.target_critic.q(
                batch_new_states, batch_target_actions)
            target_q = Variable(target_q.data, requires_grad=False)
            critic_q = self.critic.q(batch_states, batch_actions)
            
            critic_loss = self.criterion(critic_q, target_q)
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            self.move_target(self.target_actor, self.actor, TAU)
            self.move_target(self.target_critic, self.critic, TAU)

        self.running_rewards.append(ret_reward)
        if len(self.running_rewards) == WINDOW:
            print('============================')
            print('running mean: ', np.mean(self.running_rewards))
            print('============================')
            self.running_rewards = []
        return ret_reward, ret_iterations
            
                
              
            
            
        
