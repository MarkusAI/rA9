from jax import jit
from .img2col import *
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable
from jax.ops import index, index_add

class Conv2d(Function):
    id = "Conv2d"

    @staticmethod
    def forward(ctx, input, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(gamma, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)

        def np_fn(input_np, weights_np, v_current_np, gamma_np, tau_m, Vth, dt, stride=1, padding=0):
            inv_current = conv_forward(input_np, weights_np, stride, padding)

            spike_list, v_current_n = jit(jnp_fn)(x=inv_current, v_current=v_current_np,
                                                  tau_m=tau_m, Vth
                                                  =Vth, dt=dt)

            return spike_list, v_current_n, index_add(gamma_np, index[:], spike_list)

        np_args = (input.data, weights.data, v_current.data, gamma.data, tau_m, Vth, dt)
        spike, v_current_n, gamma_np = np_fn(*np_args)
        gamma.data = gamma_np
        v_current.data = v_current_n
        spike_time = jnp.multiply(spike, dt * time_step)
        spike_time = jnp.concatenate((spike, spike_time), axis=1)
        np_grad_args = (weights.data, time_step, spike_time, Vth, gamma, tau_m)
        return np_fn, np_grad_args, spike

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Conv2d, Conv2d).backward(ctx, grad_outputs)


def conv_forward(X, W, stride=1, padding=0):
    # cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = jnp.matmul(W_col, X_col)

    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = jnp.transpose(out, (3, 0, 1, 2))
    return out
