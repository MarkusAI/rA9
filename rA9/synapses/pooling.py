import jax.numpy as jnp
from rA9.networks.module import Module
from .img2col import *
from .LIF_recall import *


class pool2d(Module):
    def __init__(self, input, kernel_size, stride, tau, vth, dt, v_current):
        super(pool2d, self).__init__()
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride
        self.tau = tau
        self.vth = vth
        self.dt = dt
        self.v_current = v_current
        self.spike_list, self.v_current = LIF_recall(tau=self.tau, Vth=self.vth, dt=self.dt, x=self.input,
                                                     v_current=self.v_current)  # needto fix

    def forward(self, input, kernel_size):
        def jnp_fn(input_jnp, kernel_size, spike_list):
            return _pool_forward(input_jnp, spike_list, kernel_size)

        self.jnp_args = (input, kernel_size, self.spike_list)
        out = jnp_fn(*self.jnp_args)
        return out

    def backward(self, grad_outputs,e_grad,timestep):
        LIF_backward(self.tau, self.vth, grad_outputs, spike_list=self.spike_list, e_grad=e_grad, time=timestep)


def _pool_forward(X, spike_list, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    S_reshaped = spike_list.reshape(n * d, 1, h, w)
    S_col = im2col_indices(S_reshaped, size, size, padding=0, stride=stride)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    max_spike = jnp.argmax(S_col, axis=0)
    max_idx = jnp.argmax(X_col, axis=0)
    out = jnp.array(X_col[max_idx, range(max_idx.size)])
    out_spike = jnp.array(S_col[max_spike, range(max_idx.size)])

    out_spike = out_spike.reshape(h_out, w_out, n, d)
    out = out.reshape(h_out, w_out, n, d)
    out = jnp.matmul(out, out_spike)
    out = jnp.transpose(out, (2, 3, 0, 1))

    return out
