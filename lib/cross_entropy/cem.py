import numpy as np

class CrossEntropyAgent():

    def __init__(self, state_dim, actions_dim):
        self.state_dim = state_dim
        self.actions_dim = actions_dim

        # need to restructure code a bit if actions_dim > 2
        assert self.actions_dim == 2, 'Only two actions for now!'

        self.mu = np.zeros(self.state_dim)
        self.sigma = np.identity(self.state_dim) * 10.0
        self.all_thetas = []
        self.all_rewards = []

    # returns sample of parameters in shape (state_dim, actions_dim)
    def sample_theta(self):
        theta= np.random.multivariate_normal(self.mu, self.sigma)
        self.all_thetas.append(theta)
        return theta

    # receives reward for current theta
    def report_reward(self, reward):
        self.all_rewards.append(reward)

    # receives state and controller parameters, returns action
    def get_action(self, state, theta):
        x = np.dot(np.reshape(state, (1, 4)), theta).reshape(-1)
        return 1 if x > 0 else 0

    def refit(self):
        idx = list(range(len(self.all_thetas)))
        idx.sort(key=lambda i: self.all_rewards[i])
        idx.reverse()
        idx = idx[:int(0.1 * len(idx))]
        
        self.all_thetas = np.array(self.all_thetas)
        self.all_rewards = np.array(self.all_rewards)

        self.all_thetas = self.all_thetas[idx, :]

        self.mu = np.mean(self.all_thetas, axis=0)
        
        X = self.all_thetas
        X -= self.mu
        self.sigma = (1.0 / (X.shape[0] - 1)) * np.dot(X.transpose(), X)
        self.all_thetas = []
        self.all_rewards = []


        
