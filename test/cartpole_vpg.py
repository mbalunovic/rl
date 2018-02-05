import gym
import numpy as np
from rl.lib.policy_gradient.vpg import VanillaPolicyGradientAgent

NUM_EPISODES = 10000000000
MAX_ITERATIONS = 10**6

def main():
  env = gym.make('CartPole-v0')

  vpg_agent = VanillaPolicyGradientAgent(4, 2, 0.99, 0.01)

  for episode in range(NUM_EPISODES):
      observation = env.reset()
      rr = []

      for iteration in range(MAX_ITERATIONS):
          action = vpg_agent.get_action(observation)
          #print(action)
          #action = env.action_space.sample()
          observation, reward, done, info = env.step(action)
          # if done:
          #     reward = -200
          vpg_agent.report_reward(reward)
          
          if done:
            rr.append(iteration)
            if episode % 100 == 0:
              print('Episode {}, running it: {}'.format(
                episode,
                np.mean(rr)
              ))
              rr = []
            vpg_agent.update()
            break

if __name__ == '__main__':
    main()
