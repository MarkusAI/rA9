import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import *


class Output(Module):
    def __init__(self, out_features, tau_m=0.1, dt=1, Vth=1):
        super(Output, self).__init__()
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros(shape=(out_features, out_features)))
        self.gamma = Variable(jnp.zeros(shape=(1, out_features)))
        self.v_current = self.v_currentT(None)
        self.tau_m = tau_m
        self.time_step = 1
        self.dt = dt
        self.Vth = Vth

        self.reset_parameters()

    class v_currentT(object):
        def __init__(self, v_current):
            self.v_current = v_current

        def init_v_current(self, size):
            self.v_current = Variable(jnp.zeros(shape=size))
            return self.v_current

        def save_v_current(self, v_current):
            self.v_current = v_current
            return self.v_current

        def recall_v_current(self):
            return self.v_current

    def forward(self, input, time):
        if time == 0:
            v_current = self.v_current.init_v_current(size=(1, self.out_features))
        else:
            v_current = self.v_current.recall_v_current()
        out, v_current_ret = F.Output(input=input, weights=self.weight,
                                      v_current=v_current,
                                      tau_m=self.tau_m, dt=self.dt,
                                      time_step=time + self.time_step, Vth=self.Vth, gamma=self.gamma)

        return out

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = jnp.divide(1.0, jnp.sqrt(size[1]))
        self.weight.uniform(-stdv, stdv)
