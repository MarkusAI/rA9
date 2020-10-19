from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
import jax.numpy as jnp
from collections import OrderedDict


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, tau_m=0.1, Vth=1, dt=1, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.kernel = kernel_size
        self.weight = Parameter(jnp.zeros((out_channels, in_channels) + self.kernel_size, dtype='float32'))
        Conv2d.v_current = None
        Conv2d.gamma = None
        Conv2d.spike = None
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt
        self.stride = stride
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / jnp.sqrt(n)

        self.weight.uniform(-stdv, stdv)

    def forward(self, input, time,spikel):
        # print(self.weight.data)
        Size = (input.data.shape[0], self.out_channels,
                int(input.data.shape[2] - self.kernel + self.padding * 2 / self.stride + 1)
                , int(input.data.shape[3] - self.kernel + self.padding * 2 / self.stride + 1))

        if Conv2d.v_current is None:
            Conv2d.v_current = Variable(jnp.zeros(shape=Size, dtype='float32'))
        if Conv2d.gamma is None:
            Conv2d.gamma = Variable(jnp.zeros(shape=Size, dtype='float32'))
        if Conv2d.spike is None:
            Conv2d.spike = jnp.zeros(shape=Size)

        out = F.conv2d(input=input, time_step=time, weights=self.weight,
                       v_current=Conv2d.v_current, gamma=Conv2d.gamma,
                       tau_m=self.tau_m, Vth=self.Vth, dt=self.dt,
                       spike_list=Conv2d.spike, stride=self.stride, padding=self.padding)
        Conv2d.v_current = None
        Conv2d.gamma = None
        Conv2d.spike = None

        if spikel is None:
            spikel = OrderedDict()
            spikel.update({time: out})
        else:
            spikel.update({time: out})

        return out, time + self.dt * self.time_step,spikel

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
