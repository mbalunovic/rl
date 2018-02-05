import gym
import numpy as np
from rl.lib.cross_entropy.cem import CrossEntropyAgent

NUM_EPISODES = 10000000000
MAX_ITERATIONS = 10**6
REFIT_IT = 300

def main():
  env = gym.make('CartPole-v0')

  cem_agent = CrossEntropyAgent(4, 2)

  curr_rewards = []

  for episode in range(1, NUM_EPISODES):
      observation = env.reset()
      theta = cem_agent.sample_theta()

      total_rewards = 0

      for iteration in range(MAX_ITERATIONS):
          action = cem_agent.get_action(observation, theta)

          observation, reward, done, info = env.step(action)
          if done:
              break
          total_rewards += reward
          
          if done:
              # print('Episode {}, iterations: {}'.format(
              #     episode,
              #     iteration
              # ))
              break

      cem_agent.report_reward(total_rewards)
      curr_rewards.append(total_rewards)
      
      if episode % REFIT_IT == 0:
          print('Current rewards: ', np.mean(curr_rewards))
          curr_rewards = []
          cem_agent.refit()


if __name__ == '__main__':
    main()
