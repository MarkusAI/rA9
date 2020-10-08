import math
import numpy as jnp
from rA9.synapses.img2col import *
import jax
from rA9.networks.module import Module
from rA9.synapses.LIF_recall import *


class Linear(Module):
    def __init__(self, in_features, out_features, time, e_grad, tau, vth, dt, v_current):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.time = time
        self.e_grad = e_grad
        self.out_features = out_features
        self.zeros = jnp.zeros((out_features, in_features))
        self.tau = tau
        self.Vth = vth
        self.dt = dt
        self.v_current = v_current
        self.spike_list, self.v_current = LIF_recall(self.tau, self.Vth, self.dt, self.zeros, self.v_current)
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / math.sqrt(size[1])
        keyW = jax.random.PRNGKey(0)
        self.weight = jax.random.uniform(minval=-stdv, maxval=stdv, shape=self.weight.shape, key=keyW)

    def forward(self, input):
        def jnp_fn(input_jnp, weights_jnp, spike_list):
            out = jnp.matmul(input_jnp, weights_jnp)
            out = jnp.matmul(out, spike_list)

            return out

        jnp_args = (input, self.weight, self.spike_list)
        out = jnp_fn(*jnp_args)
        return out

    def backward(self, grad_outputs):
        LIF_backward(self.tau, self.Vth, grad_outputs, spike_list=self.spike_list, e_grad=self.e_grad, time=self.time)
