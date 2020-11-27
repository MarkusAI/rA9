import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.autograd.variable import Variable


class LIF(Module):
    gamma = None
    spike_time_list = None
    v_current = None

    def __init__(self, tau_m=100, Vth=1, dt=1):
        super(LIF, self).__init__()
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt

    def forward(self, input, time, activetime):
        if activetime == 0:
            LIF.v_current = Variable(jnp.zeros(shape=input.data.shape))
            LIF.gamma = Variable(jnp.zeros(input.data.shape))
            LIF.spike_time_list = Variable(jnp.zeros(input.data.shape))

            out, v_current, gamma, spike_time_list = F.LIF(input, LIF.v_current, self.tau_m, self.Vth, self.dt,
                                                           LIF.spike_time_list, time,
                                                           LIF.gamma)
            LIF.spike_time_list = spike_time_list
            LIF.v_current = v_current
            LIF.gamma = gamma
            
        else:
            spike_time_list = LIF.spike_time_list
            v_current = LIF.v_current
            gamma = LIF.gamma

            out, v_current, gamma, spike_time_list = F.LIF(input, v_current, self.tau_m, self.Vth, self.dt,
                                                           spike_time_list, time,
                                                           gamma)
            LIF.spike_time_list = spike_time_list
            LIF.v_current = v_current
            LIF.gamma = gamma

        return out, time + self.dt * self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
