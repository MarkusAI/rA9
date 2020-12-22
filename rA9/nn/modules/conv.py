from rA9.autograd import Function
from rA9.autograd import Variable
from jax import numpy as jnp
from .module import Module
from ..parameter import Parameter
from .. import functional as F

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size,reskey=2.0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = Parameter(jnp.zeros((out_channels, in_channels) + self.kernel_size))
        self.stride = 1
        self.padding = 0
        self.reskey = reskey
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k # Need to solve this part as JAX function
        stdv = jnp.sqrt(self.reskey/n)

        self.weight.uniform(-stdv, stdv)

    def forward(self, input):
        out = F.conv2d(input=input, weights=self.weight, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
