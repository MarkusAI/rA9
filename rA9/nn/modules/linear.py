import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
from collections import OrderedDict


class Linear(Module):
    def __init__(self, in_features, out_features, tau_m=0.1, Vth=1, dt=1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros(shape=(out_features, in_features)))
        Linear.gamma = None
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt
        self.reset_parameters()
        self.v_current = self.v_currentT(None)

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / jnp.sqrt(size[1])
        self.weight.uniform(-stdv, stdv)

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

    def forward(self, input, time, spikel):

        if time == 0:
            v_current = self.v_current.init_v_current(size=(self.out_features, input.data.shape[0]))
        else:
            v_current = self.v_current.recall_v_current()

        if Linear.gamma is None:
            Linear.gamma = Variable(jnp.zeros(shape=(self.out_features, input.data.shape[0])))

        out, v_current_ret = F.linear(input=input, time_step=time, weights=self.weight,
                                      v_current=v_current, gamma=Linear.gamma,
                                      tau_m=self.tau_m, Vth=self.Vth, dt=self.dt)

        self.v_current.save_v_current(v_current_ret)
        Linear.gamma = None

        if spikel is None:
            spikel = OrderedDict()
            spikel.update({time: out})
        else:
            spikel.update({time: out})
        return out, spikel

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
