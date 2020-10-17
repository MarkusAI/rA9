from jax import jit
from .img2col import *
from .lif import jnp_fn
from jax.ops import index_add, index
from jax.lax import index_take
from rA9.autograd import Function
from rA9.autograd import Variable
import jax.numpy as jnp
import numpy as np


# if we use this modules, we need to have a weights, so I had multipled with weights and those ones
class Pooling(Function):
    id = "Pooling"

    @staticmethod
    def forward(ctx, input, size, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt, stride=2):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)
        assert isinstance(gamma, Variable)

        def np_fn(input_np, weight, size, v_current_np, gamma_np, tau_m, Vth, dt, stride=2):
            inv_current = pool_forward(input_np, weight, size=size, stride=stride)

            spike_list, v_current_n = jit(jnp_fn)(x=inv_current, v_current=v_current_np,
                                                  tau_m=tau_m, Vth=Vth, dt=dt)
            index_add(gamma_np, index[:], spike_list)

            return spike_list, v_current_n, index_add(gamma_np, index[:], spike_list)

        np_args = (input.data, weights.data, size, v_current.data, gamma.data, tau_m, Vth, dt, stride)
        spike, v_current_n, gamma_np = np_fn(*np_args)
        gamma.data = gamma_np
        v_current.data = v_current_n
        spike_time = jnp.multiply(spike, dt * time_step)
        spike_time = jnp.concatenate((spike, spike_time), axis=1)
        np_grad_args = (weights.data, time_step, spike_time, Vth, gamma, tau_m)
        return np_fn, np_grad_args, spike

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Pooling, Pooling).backward(ctx, grad_outputs)


def pool_forward(X, W, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1
    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)

    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    X_col = np.array(X_col)
    max_idx_X = np.mean(X_col, axis=0, dtype=int)
    n_filter,v,h_filter,w_filter =W.shape
    W_col = W.reshape(n_filter,-1)

    X_col = np.matmul(W_col,X_col)
    out = np.array(X_col[max_idx_X, range(max_idx_X.size)])
    out = out.reshape(h_out, w_out, n, d)

    out = np.transpose(out, (2, 3, 0, 1))
    out = jnp.array(out)
    return out
