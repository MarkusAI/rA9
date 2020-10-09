import math
import numpy as jnp
from rA9.synapses.img2col import *
import jax
from jax.ops import index, index_add
from rA9.networks.module import Module
from rA9.synapses.LIF_recall import *


class Linear(Module):
    def __init__(self, in_features, out_features, tau=0.1, vth=1, dt=1):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.zeros = jnp.zeros((out_features,in_features))
        self.tau = tau
        self.weight=None
        self.Vth = vth
        self.dt = dt
        self.v_current = jnp.zeros((out_features,in_features))
        self.spike_list, self.v_current = LIF_recall(self.tau, self.Vth, self.dt, self.zeros, self.v_current)
        self.reset_parameters()
        if type(self.in_features) == tuple:
            self.gamma = jnp.zeros(shape=(in_features,in_features))
        else:
            self.gamma = jnp.zeros(shape=in_features)

    def reset_parameters(self):
        size = self.zeros.shape
        stdv = 1. / math.sqrt(size[1])
        keyW = jax.random.PRNGKey(0)
        self.weight = jax.random.uniform(minval=-stdv, maxval=stdv, shape=self.zeros.shape, key=keyW)

    def forward(self, input):
        def jnp_fn(input_jnp, weights_jnp):
            out = jnp.matmul(input_jnp, weights_jnp)

            return out

        jnp_args = (input, self.weight)
        if type(self.in_features) == tuple:
            index_add(self.gamma,index[:,:], input)
        else:
            index_add(self.gamma,index[:], input)

        return jnp_fn(*jnp_args)


    def backward(self, e_grad, timestep):
        fn, c, fh, fw = self.weight.shape
        tau = jnp.divide(jnp.subtract(timestep, self.spike_list[1]), -self.tau)
        prime = jnp.multiply(jnp.exp(tau), (-1 / self.tau))
        aLIFnet = jnp.multiply(1 / self.Vth, (1 + jnp.multiply(jnp.divide(1, self.gamma, prime))))
        self.grad = jnp.multiply(self.weight, aLIFnet)

        self.grad = self.grad.transpose(1, 0).reshape(fn, c, fh, fw)

        d_col = jnp.multiply(self.weight, aLIFnet)
        dx = col2im_indices(d_col, self.con.shape, fh, fw)

        return LIF_backward(self.tau, self.Vth, dx,
                            spike_list=self.spike_list, e_grad=e_grad,
                            time=timestep), self.weight
