import gym
import numpy as np
from rl.lib.evolution.es import EvolutionStrategyAgent

NUM_EPISODES = 10000000000
MAX_ITERATIONS = 10**6

def get_params_dim():
    return 100 * 6400 + 1 * 100

# should add biases, but yolo 
def get_params(params):
    w1 = params[:100 * 6400].reshape(100, 6400)
    w2 = params[100 * 6400:].reshape(1, 100)
    return w1, w2

def forward(x, params):
    x = x.reshape(6400, 1)
    w1, w2 = get_params(params)
    x = np.dot(w1, x)
    x[x < 0] = 0
    x = np.dot(w2, x)
    x = 1.0 / (1.0 + np.exp(-x))
    return x

# preprocess and downsample, copied from Karpathy
def prepro(I):
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def main():
  env = gym.make('Pong-v0')

  param_dim = get_params_dim()
  
  params = np.zeros(param_dim)
  
  es_agent = EvolutionStrategyAgent(0.01, 1, param_dim, 0.99)
  es_agent.init_params(params)

  n_samples = 20

  rr = []
  for episode in range(NUM_EPISODES):
      rews = []
      
      for i in range(n_samples):
          params = es_agent.get_params()
          observation = env.reset()
          observation = prepro(observation)
          
          for iteration in range(MAX_ITERATIONS):
              sigma = forward(observation, params)
              action = 1 if sigma[0, 0] > 0.5 else 0
          
              observation, reward, done, info = env.step(action)
              observation = prepro(observation)
              
              if done:
                  es_agent.report_return(reward)
                  rews.append(reward)
                  print(reward)
                  break

      rr.append(np.mean(rews))
      if episode % 100 == 0:
          print('Episode {}, running it: {}'.format(
              episode,
              np.mean(rr)
          ))
          rr = []
      print(episode, np.mean(rews))
      es_agent.update()
     

if __name__ == '__main__':
    main()
