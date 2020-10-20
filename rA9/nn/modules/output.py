import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter


class Output(Module):
    def __init__(self, out_features, tau_m=0.1, dt=1,Vth=1):
        super(Output, self).__init__()
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros(shape=(out_features, out_features)))
        self.gamma = Parameter(jnp.zeros(shape=(1, out_features)))
        self.v_current = Parameter(jnp.zeros(shape=(1, out_features)))
        self.tau_m = tau_m
        self.time_step = 1
        self.dt = dt
        self.Vth = Vth

        self.reset_parameters()

    def forward(self, input, time):
        return F.Output(input=input, weights=self.weight,
                        v_current=self.v_current,
                        tau_m=self.tau_m, dt=self.dt,
                        time_step=time+self.time_step,Vth=self.Vth,gamma=self.gamma)

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / jnp.sqrt(size[1])
        self.weight.uniform(-stdv, stdv)
