from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
import jax.numpy as jnp


class LIF(Module):
    def __init__(self, tau_m=100, Vth=1, dt=1):
        super(LIF, self).__init__()
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt
        self.gamma = self.including_object(None)
        self.spike_time_list = self.including_object(None)
        self.v_current = self.including_object(None)

    class including_object(object):
        def __init__(self, data):
            self.data = data

        def init_data(self, size):
            self.data = Variable(jnp.zeros(shape=size))
            return self.data

        def save_data(self, new_data):
            self.data = new_data
            return self.data

        def recall_data(self):
            return self.data

    def forward(self, input, time, activetime):

        if activetime == 0:
            v_current = self.v_current.init_data(input.data.shape)
            gamma = self.gamma.init_data(input.data.shape)
            spike_time_list = self.spike_time_list.init_data(input.data.shape)
        else:
            v_current = self.v_current.recall_data()
            gamma = self.gamma.recall_data()
            spike_time_list = self.spike_time_list.recall_data()

        out, v_current_ret, gamma_t ,spike_time_list_ret = F.LIF(input, v_current, self.tau_m, self.Vth, self.dt, spike_time_list, time,
                                               gamma)


        self.gamma.save_data(gamma_t)

        self.v_current.save_data(v_current_ret)

        self.spike_time_list.save_data(spike_time_list_ret)

        return out , time+self.dt*self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
