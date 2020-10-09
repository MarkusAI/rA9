import math
import numpy as jnp
from rA9.synapses.img2col import *
from rA9.synapses.LIF_recall import *
from rA9.networks.module import Module
from jax import random

class Conv2d(Module):

    def __init__(self, input_channels,output_channels, kernel_size,tau=0.1,vth=1,dt=1,  stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.grad =None
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.zeros = jnp.zeros((output_channels, input_channels) + self.kernel_size)
        self.tau= tau
        self.Vth= vth
        self.dt= dt
        self.S_col=None
        self.spike_list = jnp.zeros(kernel_size,kernel_size)
        self.v_current = jnp.zeros(kernel_size,kernel_size)
        self.reset_parameters()
        self.con =None

    def reset_parameters(self):
        n = self.input_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        keyW = random.PRNGKey(0)

        self.weight = random.uniform(minval=-stdv, maxval=stdv, shape=self.zeros.shape, key=keyW)

    def forward(self, input):

        def jnp_fn(input,weights_jnp, stride=1, padding=0):
            self.con =input
            n_filters, d_filter, h_filter, w_filter = weights_jnp.shape
            n_x, d_x, h_x, w_x = input.shape
            h_out = (h_x - h_filter + 2 * padding) / stride + 1
            w_out = (w_x - w_filter + 2 * padding) / stride + 1

            if not h_out.is_integer() or not w_out.is_integer():
                raise Exception('Invalid output dimension!')  # 오류 체크
            h_out, w_out = int(h_out), int(w_out)
            W_col = weights_jnp.reshape(n_filters, -1)
            self.S_col = im2col_indices(input,h_filter,w_filter,padding=padding,stride=stride)


            out= jnp.matmul(W_col,S_col)

            out = out.reshape(n_filters, h_out, w_out, n_x)
            out = jnp.transpose(out, (3, 0, 1, 2))

            return out

        output = jnp_fn(input,self.weight)

        args= jnp.array([input,self.weight])#dd

        return output ,args


    def backward(self, e_grad,timestep):
       fn,c,fh,fw = self.weight.shape
       gamma  = self.spike_list[0]

       tau= jnp.divide(jnp.subtract(timestep,self.spike_list[1]),-self.tau)
       prime= jnp.multiply(jnp.exp(tau),(-1/self.tau))
       aLIFnet= jnp.multiply(1/self.Vth,(1+jnp.multiply(jnp.divide(1,gamma,prime))))
       self.grad= jnp.multiply(self.weight,aLIFnet)

       self.grad= self.grad.transpose(1,0).reshape(fn,c,fh,fw)

       d_col= jnp.multiply(self.weight,aLIFnet)
       dx=col2im_indices(d_col,self.con.shape,fh,fw)

       return LIF_backward(self.tau,self.Vth,dx,
                            spike_list=self.spike_list,e_grad=e_grad,
                            time=timestep), self.weight
