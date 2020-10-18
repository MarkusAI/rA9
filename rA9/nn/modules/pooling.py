from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter

import jax.numpy as jnp


class Pooling(Module):

    def __init__(self, channel, size=2, stride=2, tau_m=0.1, Vth=1, dt=1):
        super(Pooling, self).__init__()
        self.size = size
        self.stride = stride
        self.kernel =(size,size)

        self.weight = Parameter(jnp.ones((channel,1)+self.kernel))
        Pooling.v_current = None
        Pooling.gamma = None
        Pooling.spike_list = None
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt

    def forward(self, input, time):
        insize = input.data
        #print(self.weight.data)
        Size = (insize.shape[0], insize.shape[1],
                int((insize.shape[2] - self.size) / self.stride + 1),
                int((insize.shape[3] - self.size) / self.stride + 1))

        if Pooling.v_current is None:
            Pooling.v_current = Parameter(jnp.zeros(shape=Size))
        if Pooling.gamma is None:
            Pooling.gamma = Parameter(jnp.zeros(shape=Size))
        if Pooling.spike_list is None:
            Pooling.spike_list = Parameter(jnp.zeros(shape=Size))
        out = F.pooling(input=input, size=self.size, time_step=time,
                        weights=self.weight, spike_list=Pooling.spike_list,
                        v_current=Pooling.v_current, gamma=Pooling.gamma, tau_m=self.tau_m,
                        Vth=self.Vth, dt=self.dt, stride=self.stride)
        Pooling.gamma = None
        Pooling.spike_list = None
        Pooling.v_current = None

        return out, time + self.dt * self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
