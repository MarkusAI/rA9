import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.autograd.variable import Variable


class Output(Module):
    v_current = None

    def __init__(self, out_features, tau_m=0.1, dt=1, Vth=1):
        super(Output, self).__init__()
        self.out_features = out_features
        self.tau_m = tau_m
        self.time_step = 1
        self.Vth = Vth
        self.dt = dt

        self.reset_parameters()

    def forward(self, input, time, activetime):
        if activetime == 0:
            Output.v_current = Variable(jnp.zeros(shape=(1, self.out_features)))
            out, v_current_ret = F.Output(input=input,
                                          v_current=Output.v_current,
                                          tau_m=self.tau_m, dt=self.dt,
                                          time_step=time + self.time_step)
            Output.v_current = v_current_ret
        else:
            out, v_current_ret = F.Output(input=input,
                                          v_current=Output.v_current,
                                          tau_m=self.tau_m, dt=self.dt,
                                          time_step=time + self.time_step)
            Output.v_current = v_current_ret

        return out, time + self.dt * self.time_step

    def reset_parameters(self):
        pass
