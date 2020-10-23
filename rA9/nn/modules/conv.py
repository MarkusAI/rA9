import jax.numpy as np
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter

class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(np.zeros((out_channels, in_channels) + self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / np.sqrt(n)
        self.weight.uniform(-stdv, stdv)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.stride, self.padding)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

