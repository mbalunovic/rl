"""
Implementation based on paper "Evolution Strategies as a
Scalable Alternative to Reinforcement Learning"
https://arxiv.org/pdf/1703.03864.pdf
"""

import numpy as np

class EvolutionStrategyAgent():
    def __init__(self, lr, std, param_dim, lr_decay=1.0):
        self.lr = lr
        self.std = std
        self.param_dim = param_dim
        self.grads = []
        self.lr_decay = lr_decay

    def init_params(self, theta):
        assert(theta.shape[0] == self.param_dim)
        self.theta = theta

    def get_params(self):
        self.eps = np.random.normal(0, 1, self.param_dim)
        return self.theta + self.std * self.eps

    def report_return(self, f):
        sample_grad = f * self.eps / self.std
        self.grads.append(sample_grad)

    def update(self):
        grad_estimate = np.mean(self.grads, axis=0)
        self.theta += self.lr * grad_estimate
        self.grads = []
        self.lr *= self.lr_decay



    

        
