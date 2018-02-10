import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

"""
Neural network outputs multidimensional continuous action.
Action can also be bounded between -1 and 1 (just applies tanh
on last layer).
"""
class NNContinuousPolicy(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, bound=False):
        super(NNContinuousPolicy, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.bound = bound

        assert(len(hidden_dims) >= 1)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.params = 0
        for param in self.parameters():
            self.params += param.data.view(-1).size()[0]

    # returns total number of free parameters of neural network
    def total_params(self):
        return self.params

    # sets neural net parameters to those contained in theta
    def set_params(self, theta):
        assert(theta.shape[0] == self.params)
        i = 0
        for param in self.parameters():
            k = param.data.view(-1).size()[0]
            param.data = torch.from_numpy(theta[i:i+k].reshape(param.size()))

    # receives x and calculates output of neural net on input x 
    def forward(self, x):
        x = torch.from_numpy(x)
        x = x.type(torch.FloatTensor)
        x = Variable(x.view(1, -1))

        first = True
        for i, fc in enumerate(self.layers):
            if not first:
                x = F.relu(x)
            first = False
            x = fc(x)
            
        if self.bound:
            x = F.tanh(x)
        return x

    # moves neural net parameters in the direction of the gradient
    def update(learning_rate):
        for f in self.parameters():
            f.data.add_(f.grad.data * self.learning_rate)
        self.zero_grad()
    
        
        
