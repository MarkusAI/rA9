from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter

import jax.numpy as jnp


class Pooling(Module):

    def __init__(self, input, size,dt, Vth, tau_m, stride=1):
        super(Pooling, self).__init__()
        self.stride = stride
        self.input= input
        self.dt=dt
        self.time_step =1
        self.Vth =Vth
        self.tau_m =tau_m
        self.weight = jnp.zeros(jnp.shape(self.input.data))
        self.size = size
        self.v_current =None
        self.reset_parameters()

    def reset_parameters(self):
        n = self.input.data
        for k in self.kernel_size:
            n *= k
        stdv = 1. / jnp.sqrt(n)
        self.weight.uniform(-stdv, stdv)
           
    def forward(self, input,time):
        if self.v_current is None:
            self.v_current = Parameter(jnp.zeros(jnp.shape((input-self.kernel)//self.stride+1)))
        if self.gamma is None:
            self.v_current = Parameter(jnp.zeros(jnp.shape((input-self.kernel_size)//self.stride+1)))
        return F.pooling(input=self.input,weight=self.weight,v_current=self.v_current,
                         time_step=time,Vth=self.Vth,size=self.size,stride=self.stride,
                         dt=self.dt,tau_m=self.tau_m),time+self.dt*self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'