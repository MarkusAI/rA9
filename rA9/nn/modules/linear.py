import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, tau_m=0.1, Vth=1, dt=1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros(shape=(out_features, in_features)))
        self.v_current = Parameter(jnp.zeros(shape=(1, in_features)))
        self.gamma = Parameter(jnp.zeros(shape=(1, in_features)))
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / jnp.sqrt(size[1])
        self.weight.uniform(-stdv, stdv)

    def forward(self, input, time):
        return F.linear(input=input, weight=self.weight,
                        v_current=self.v_current, gamma=self.gamma,
                        tau_m=self.tau_m, Vth=self.Vth, dt=self.dt
                        ), time + self.dt*self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
