import numpy as np

class CrossEntropyNeuralAgent():

    def __init__(self, state_dim, actions_dim):
        self.state_dim = state_dim
        self.actions_dim = actions_dim
        self.param_dim = self.get_param_dim()

        self.mu = np.zeros(self.param_dim)
        self.sigma = np.identity(self.param_dim) * 10.0
        self.all_thetas = []
        self.all_rewards = []
        
        self.alpha = 1.20

    def get_param_dim(self):
        return 41

    def get_weights(self, param):
        assert(param.shape[0] == self.param_dim)
        w1 = param[:12].reshape((4, 3))
        b1 = param[12:16].reshape((4, 1))
        w2 = param[16:32].reshape((4, 4))
        b2 = param[32:36].reshape((4, 1))
        w3 = param[36:40].reshape((1, 4))
        b3 = param[40:41].reshape((1, 1))
        return w1, w2, w3, b1, b2, b3

    # returns sample of parameters in shape (state_dim, actions_dim)
    def sample_theta(self):
        theta = np.random.multivariate_normal(self.mu, self.sigma * self.alpha)
        self.all_thetas.append(theta)
        return theta

    # returns mean parameters
    def get_mean_theta(self):
        return self.mu

    # receives reward for current theta
    def report_reward(self, reward):
        self.all_rewards.append(reward)

    # receives state and controller parameters, returns action
    def get_action(self, state, theta):
        x = np.dot(np.reshape(state, (1, self.state_dim)), theta)
        x = x.reshape(-1)
        # only works with 2 discrete actions, fix this
        return 1 if x > 0 else 0

    # receives state and controller parameters, returns
    # continuous action between -1 and 1
    def get_action_cont(self, state, theta):
        #x = np.dot(np.reshape(state, (1, self.state_dim)), theta)
        w1, w2, w3, b1, b2, b3 = self.get_weights(theta)
        x = np.reshape(state, (self.state_dim, 1))
        x = np.tanh(np.dot(w1, x) + b1)
        x = np.tanh(np.dot(w2, x) + b2)
        x = np.tanh(np.dot(w3, x) + b3)
        x = x.reshape(-1)
        return x

    def refit(self):
        idx = list(range(len(self.all_thetas)))
        idx.sort(key=lambda i: self.all_rewards[i])
        idx.reverse()
        idx = idx[:int(0.01 * len(idx))]

        self.all_thetas = np.array(self.all_thetas)
        self.all_rewards = np.array(self.all_rewards)

        self.all_thetas = self.all_thetas[idx, :]

        self.mu = np.mean(self.all_thetas, axis=0)
        self.sigma = np.zeros((self.param_dim, self.param_dim))

        for i in range(self.param_dim):
            self.sigma[i,i] = self.all_thetas[:, i].std()
        
        self.all_thetas = []
        self.all_rewards = []

        self.alpha *= 0.99

        
