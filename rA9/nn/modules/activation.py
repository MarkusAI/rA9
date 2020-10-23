from jax import numpy as np
from .module import Module
from .. import functional as F

class LIF(Module):

    def __init__(self,tau_m=0.1,Vth=1,dt=1):
        super(LIF, self).__init__()
        self.s_time_list = None
        self.v_current = 0
        self.gamma = 0
        self.tau_m = tau_m
        self.Vth = Vth
        self.dt = dt

    def forward(self, input, time):
        if self.s_time_list is None:
            self.s_time_list = np.zeros(input.data.shape)

        output = F.LIF(input=input,v_current=self.v_current,
                     tau_m=self.tau_m, Vth=self.Vth, dt=self.dt,
                     s_time_list=self.s_time_list,time=time,
                     gamma=self.gamma)
        #spike time list to output
        print(len(output.data))
        self.v_current = self.v_current + v_current #Because of the problem which JAX has.
        self.gamma = self.gamma + spike #Because of the problem which JAX has.
        self.s_time_list = s_time_list
        return grad_fn, time+1