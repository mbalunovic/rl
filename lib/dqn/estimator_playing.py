import argparse
import sys
import gym
import tensorflow as tf
import numpy as np
import random

from collections import deque

ACTIONS_DIM = 2
OBSERVATIONS_DIM = 4

FLAGS = None

def model_fn(features, labels, mode, params):
  # First hidden layer
  hidden1 = tf.layers.dense(
    features['x'],
    10,
    activation=tf.nn.relu,
  )

  # Output layer
  output_layer = tf.layers.dense(
    hidden1,
    ACTIONS_DIM,
  )

  predictions = tf.reshape(output_layer, [-1, ACTIONS_DIM])

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
    )

  loss = tf.losses.mean_pairwise_squared_error(labels, predictions)

  eval_metric_ops = {
    'rmse': tf.metrics.root_mean_squared_error(
      tf.cast(labels, tf.float64),
      predictions,
    )
  }

  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=params['learning_rate'],
  )
  train_op = optimizer.minimize(
    loss=loss,
    global_step=tf.train.get_global_step(),
  )

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops,
  )

class ReplayBuffer():

  def __init__(self, max_size):
    self.max_size = max_size
    self.transitions = deque()

  def add(self, observation, labels):
    if len(self.transitions) > self.max_size:
      self.transitions.popleft()
    self.transitions.append((observation, labels))

  def sample(self):
    return random.sample(self.transitions)



def get_q(model, observation):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np_obs},
    num_epochs=1,
    shuffle=False,
  )
  predictions = list(model.predict(input_fn=predict_input_fn))
  return predictions[0]

def train(model, observation, labels):
  np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np_obs},
    y=np.array([labels]),
    shuffle=False,
  )
  model.train(input_fn=train_input_fn,
              steps=200)

def main(unused_args):
  model_params = {
    'learning_rate': 0.1,
  }

  action_model = tf.estimator.Estimator(
    model_fn=model_fn,
    params=model_params,
  )

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={},
    y=np.array([]),
    shuffle=False,
  )
  action_model.train(input_fn=train_input_fn,
              steps=200)


  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array([[1.0, 1.0, 1.0, 1.0]])},
    num_epochs=1,
    shuffle=False,
  )

  predictions = list(action_model.predict(input_fn=predict_input_fn))
  print predictions[0]

  exit(0)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': np.array([[1.0, 1.0, 1.0, 1.0]])},
    y=np.array([[1.0, 1.0]]),
    shuffle=False,
  )
  action_model.train(input_fn=train_input_fn,
              steps=200)

  for i in range(1000):
    #env.render()

    q_values = get_q(action_model, observation)
    labels = q_values.copy()

    old_observation = observation
    action = np.argmax(q_values)
    observation, reward, done, info = env.step(action)

    labels[action] = reward
    if not done:
      q_values2 = get_q(action_model, observation)
      action2 = np.argmax(q_values2)
      labels[action] += GAMMA * q_values2[action2]
    replay.add(old_observation, labels)

    if done:
      print 'Game finished after {} iterations'.format(i)
      break

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

