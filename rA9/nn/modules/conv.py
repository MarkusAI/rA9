from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter

import jax.numpy as jnp


class Conv2d(Module):

    def __init__(self, in_channels, out_channels, kernel_size, dt, Vth, tau_m, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.dt = dt
        self.Vth = Vth
        self.gamma = None
        self.time_step = 1
        self.tau_m = tau_m
        self.stride = stride
        self.v_current = None
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = Parameter(jnp.zeros((out_channels, in_channels) + self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / jnp.sqrt(n)
        self.weight.uniform(-stdv, stdv)

    def forward(self, input, time):
        if self.v_current is None:
            self.v_current = Parameter(jnp.zeros(jnp.shape((input-self.kernel_size)//self.stride+1)))
        if self.gamma is None:
            self.v_current = Parameter(jnp.zeros(jnp.shape((input-self.kernel_size)//self.stride+1)))

        return F.conv2d(input=input, weights=self.weight,
                        stride=self.stride, padding=self.padding,
                        v_current=self.v_current, Vth=self.Vth,
                        tau_m=self.tau_m,time_step=time,dt=self.dt), time + self.dt*self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
