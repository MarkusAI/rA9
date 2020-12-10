import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.autograd.variable import Variable


class LIF(Module):

    def __init__(self, tau_m=100, Vth=1, dt=1):
        super(LIF, self).__init__()
        self.spike_time_list = None
        self.v_current = None
        self.gamma = None
        self.tau_m = tau_m
        self.time_step = 1
        self.Vth = Vth
        self.dt = dt

    def forward(self, input, time):
        if time == 0:

            self.spike_time_list = None
            self.v_current = None
            self.gamma = None

            self.v_current = Variable(jnp.zeros(shape=input.data.shape))
            self.gamma = Variable(jnp.zeros(shape=input.data.shape))
            self.spike_time_list = Variable(jnp.zeros(shape=input.data.shape))

        out, v_current, gamma, spike_time_list = F.LIF(input, self.v_current, self.tau_m, self.Vth, self.dt,
                                                       self.spike_time_list, time + 1,
                                                       self.gamma)
        self.spike_time_list = spike_time_list
        self.v_current = v_current
        self.gamma = gamma

        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'