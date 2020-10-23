import math

import rA9
from rA9.autograd import Variable

from .module import Module
from rA9.nn.parameter import Parameter
from .. import functional as F

import jax.numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.zeros((out_features, in_features)))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / math.sqrt(size[1])
        self.weight.uniform(-stdv, stdv)

    def forward(self, input, time):
        return F.linear(input, self.weight), time

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'