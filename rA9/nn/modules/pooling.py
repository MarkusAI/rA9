from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter

import jax.numpy as jnp


class Pooling(Module):

    def __init__(self, input, size, stride=1,tau_m=0.1, Vth=1, dt=1):
        super(Pooling, self).__init__()
        self.input= input
        self.size = size
        self.stride=stride
        self.weight = Parameter(jnp.zeros(shape=(jnp.shape(input))))
        self.v_current = None
        self.gamma = None
        self.spike_list = None
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt
        self.reset_parameters()

    def forward(self, input):
        insize=input.data
        if self.v_current is None:
            self.v_current = Parameter(jnp.zeros(jnp.shape(int(insize.shape[0]/self.size+1),int(insize.shape[1]/self.size+1))))
        if self.gamma is None:
            self.gamma = Parameter(jnp.zeros(jnp.shape((int(insize.shape[0]/self.size+1),int(insize.shape[1]/self.size+1)))))
        if self.spike_list is None:
            self.spike_list = jnp.zeros(jnp.shape((int(insize.shape[0]/self.size+1),int(insize.shape[1]/self.size+1))))
        return F.pooling(input=self.input,size=self.size,time_step=self.time_step,
                         weights=self.weight,spike_list=self.spike_list,
                         v_current=self.v_current,gamma=self.gamma,tau_m=self.tau_m,
                         Vth=self.Vth,dt=self.dt,stride=self.stride)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'