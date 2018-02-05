import gym
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4

class VanillaPolicyGradient(object):

  def __init__(self):
    self.model = Sequential()
    self.model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    self.model.add(Dense(16, input_shape=(OBSERVATIONS_DIM,), activation='relu'))
    self.model.add(Dense(2, activation='softmax'))

  def get_train_fn(self):
    pass

  def get_action(self, state):
    action_probs = self.model.predict(state)
    return np.random.choice(
      range(ACTIONS_DIM),
      p=action_probs.flatten(),
    )

  def fit(self, states, actions, rewards):
    pass

  def discount_rewards(self, rewards):
    pass

def run_episode(env, agent):
  states, actions, rewards = [], [], []

  observation = env.reset()
  state = np.reshape(observation, [-1, OBSERVATIONS_DIM])

  while True:
    action = agent.get_action(state)

    new_observation, reward, done, info = env.step(action)
    new_state = np.reshape(new_observation, [-1, OBSERVATIONS_DIM])

    if done:
      reward = -200

    states.append(state)
    actions.append(action)
    rewards.append(reward)

    if done:
      agent.fit(states, actions, rewards)

    state = new_state




def main():
  agent = VanillaPolicyGradient()

  env = gym.make('CartPole-v0')
  observation = env.reset()
  state = np.reshape(observation, [-1, OBSERVATIONS_DIM])

  print agent.get_action(state)

if __name__ == '__main__':
  main()
