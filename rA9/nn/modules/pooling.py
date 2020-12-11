from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
import jax.numpy as jnp
from collections import OrderedDict


class Pooling(Module):

    def __init__(self, channel, size=2, stride=2, staticweight=0.25):
        super(Pooling, self).__init__()
        self.size = size
        self.stride = stride
        self.kernel = (size, size)
        self.weight = Variable(jnp.full((channel, 1) + self.kernel, staticweight))

    def forward(self, input):
        out = F.pooling(input=input, size=self.size, weights=self.weight, stride=self.stride)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
