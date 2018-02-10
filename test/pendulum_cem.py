import gym
import numpy as np
from rl.lib.cross_entropy.cem_neural import CrossEntropyNeuralAgent

NUM_EPISODES = 10000000000
MAX_ITERATIONS = 100000000
REFIT_IT = 1000
K = 1000000
REPEAT = 10

def main():
  env = gym.make('Pendulum-v0')

  cem_agent = CrossEntropyNeuralAgent(3, 1)

  curr_rewards = []

  for episode in range(1, NUM_EPISODES):
      total_rewards = 0
      if episode % 1000 == 1:
        theta = cem_agent.get_mean_theta()
      else:
        theta = cem_agent.sample_theta()

      for r in range(REPEAT):
        observation = env.reset()
        if episode % K == 0:
          env.render()

        rep_rewards = 0
        for iteration in range(MAX_ITERATIONS):
          action = 2 * cem_agent.get_action_cont(observation, theta)

          observation, reward, done, info = env.step(action)
          rep_rewards += reward
          
          if episode % K == 0:
            env.render()
          if done:
            #print(rep_rewards)
            total_rewards += rep_rewards
            break

      total_rewards /= REPEAT

      if episode % 1000 != 1:
        cem_agent.report_reward(total_rewards)
        curr_rewards.append(total_rewards)
      else:
        print('Reward of mean: ', total_rewards)
      
      if episode % REFIT_IT == 0:
          print('Current rewards: ', np.mean(curr_rewards))
          curr_rewards = []
          cem_agent.refit()


if __name__ == '__main__':
    main()
