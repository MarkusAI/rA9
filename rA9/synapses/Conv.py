import math
import numpy as jnp
from rA9.synapses.img2col import *
from rA9.synapses.LIF_recall import LIF_recall
from rA9.networks.module import Module
from jax import random

class Conv2d(Module):

    def __init__(self, input_channels,tau,vth,dt,v_current, output_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.zeros = jnp.zeros((output_channels, input_channels) + self.kernel_size)
        self.tau= tau
        self.Vth= vth
        self.dt= dt
        self.v_current=v_current

        self.spike_list, self.v_current = LIF_recall(tau=self.tau,Vth=self.Vth,dt=self.dt,x=self.zeros,v_current=self.v_current)  # needto fix

        self.reset_parameters()

    def reset_parameters(self):
        n = self.input_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        keyW = random.PRNGKey(0)

        self.weight = random.uniform(minval=-stdv, maxval=stdv, shape=self.zeros.shape, key=keyW)

    def forward(self, input):

        def jnp_fn(input_jnp, weights_jnp, spike_list, stride=1, padding=0):

            n_filters, d_filter, h_filter, w_filter = weights_jnp.shape
            n_x, d_x, h_x, w_x = input_jnp.shape
            h_out = (h_x - h_filter + 2 * padding) / stride + 1
            w_out = (w_x - w_filter + 2 * padding) / stride + 1

            if not h_out.is_integer() or not w_out.is_integer():
                raise Exception('Invalid output dimension!')  # 오류 체크
            h_out, w_out = int(h_out), int(w_out)
            X_col = im2col_indices(input_jnp, h_filter, w_filter, padding=padding, stride=stride)
            W_col = weights_jnp.reshape(n_filters, -1)
            S_col = spike_list.reshape(n_filters,-1)

            out = jnp.matmul(W_col, X_col)
            out= jnp.matmul(out,S_col)

            out = out.reshape(n_filters, h_out, w_out, n_x)
            out = jnp.transpose(out, (3, 0, 1, 2))

            return out

        output = jnp_fn(input,self.weight,self.spike_list)

        return output,self.v_current

    def backward(self, grad_outputs):
