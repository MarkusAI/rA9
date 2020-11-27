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

    def forward(self, input, time, activetime):
        if activetime == 0:
            self.v_current = Variable(jnp.zeros(shape=input.data.shape))
            self.gamma = Variable(jnp.zeros(input.data.shape))
            self.spike_time_list = Variable(jnp.zeros(input.data.shape))

            out, v_current, gamma, spike_time_list = F.LIF(input, LIF.v_current, self.tau_m, self.Vth, self.dt,
                                                           LIF.spike_time_list, time,
                                                           LIF.gamma)
            self.spike_time_list = spike_time_list
            self.v_current = v_current
            self.gamma = gamma
            
        else:
            spike_time_list = self.spike_time_list
            v_current = self.v_current
            gamma = self.gamma

            out, v_current, gamma, spike_time_list = F.LIF(input, v_current, self.tau_m, self.Vth, self.dt,
                                                           spike_time_list, time,
                                                           gamma)
            self.spike_time_list = spike_time_list
            self.v_current = v_current
            self.gamma = gamma

        return out, time + self.dt * self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
