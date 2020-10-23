from rA9.autograd import Function
from rA9.autograd import Variable
from jax import numpy as np
from jax import grad


class Conv2d(Function):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = Parameter(jnp.zeros((out_channels, in_channels) + self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.reset_parameters()

    def forward(self, input):
        out = F.conv2d(input=input, weights=self.weight, stride=self.stride, padding=self.padding)

        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'





