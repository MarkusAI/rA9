import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
from ..spike import Spike
class Linear(Module):
    def __init__(self, in_features, out_features, tau_m=0.1, Vth=1, dt=1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros(shape=(out_features, in_features)))
        self.spike = Spike(jnp.zeros(shape=(out_features,in_features)))

        Linear.v_current = None
        Linear.gamma = None
        Linear.spike_list = None
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
        #print(self.weight.data)
        
        if Linear.gamma is None:
            Linear.gamma = Variable(jnp.zeros(shape=(self.out_features, input.data.shape[0])))
        if Linear.v_current is None:
            Linear.v_current = Variable(jnp.zeros(shape=(self.out_features, input.data.shape[0])))
        out = F.linear(input=input, time_step=time, weights=self.weight,
                       v_current=Linear.v_current, gamma=Linear.gamma,
                       tau_m=self.tau_m, Vth=self.Vth, dt=self.dt,
                       spike_list=self.spike)
        
        Linear.gamma=None
        Linear.v_current=None
        return out, time + self.dt * self.time_step

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
