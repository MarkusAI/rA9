import math

import bintorch
from bintorch.autograd import Variable

from ..networks.module import Module
from Markus.networks.parameter import Parameter
from ._functions import *

import jax.numpy as jnp


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros((out_features, in_features)))
        if bias:
            self.bias = Parameter(jnp.zeros((out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / math.sqrt(size[1])
        self.weight.uniform(-stdv, stdv)
        if self.bias is not None:
            self.bias.uniform(-stdv, stdv)

    def forward(self, input):
        return linear.apply(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
