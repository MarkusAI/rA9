from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter

import jax.numpy as jnp


class Pooling(Module):

    def __init__(self, input, size, stride=1):
        super(Pooling, self).__init__()
        self.stride = stride
        self.input= input
        self.size = size

    def forward(self, input):
        return F.pooling(input=self.input,size=self.size,stride=self.stride)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'