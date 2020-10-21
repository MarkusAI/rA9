from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
import jax.numpy as jnp
from collections import OrderedDict


class Pooling(Module):

    def __init__(self, channel, size=2, stride=2, tau_m=0.1, Vth=1, dt=1,staticweight=0.25):
        super(Pooling, self).__init__()
        self.size = size
        self.stride = stride
        self.kernel = (size, size)

        self.weight = Variable(jnp.full((channel, 1) + self.kernel,staticweight))
        Pooling.v_current = None
        Pooling.gamma = None
        Pooling.spike = None
        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt
        self.v_current = self.v_currentT(None)

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
        insize = input.data
        # print(self.weight.data)
        Size = (insize.shape[0], insize.shape[1],
                int((insize.shape[2] - self.size) / self.stride + 1),
                int((insize.shape[3] - self.size) / self.stride + 1))

        if time == 0:
            v_current = self.v_current.init_v_current(Size)
        else:
            v_current = self.v_current.recall_v_current()

        if Pooling.gamma is None:
            Pooling.gamma = Variable(jnp.zeros(shape=Size))

        out, v_current_ret = F.pooling(input=input, size=self.size, time_step=time,
                                       weights=self.weight, v_current=v_current, gamma=Pooling.gamma, tau_m=self.tau_m,
                                       Vth=self.Vth, dt=self.dt, stride=self.stride)
        self.v_current.save_v_current(v_current_ret)
        Pooling.gamma = None

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
