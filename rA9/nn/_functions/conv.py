from jax import jit
from .img2col import *
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable


class Conv2d(Function):
    id = "Conv2d"
    @staticmethod
    def forward(ctx, input, weights, v_current,time_step, tau_m, Vth, dt, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)

        def np_fn(input_np, weights_np, stride=1, padding=0):
            return conv_forward(X=input_np,W= weights_np, stride=stride,padding= padding)

        np_args = (input.data, weights.data,stride,padding)
        inv_current = np_fn(*np_args)
        spike_list, v_current = jit(jnp_fn)(x=inv_current,
                                            v_current=v_current.data,
                                            tau_m=tau_m.data,
                                            Vth=Vth.data,
                                            dt=dt.data)
        np_args = (input.data, weights.data, v_current.data, gamma.data, tau_m, Vth, dt)
        grad_np_args = (weights.data, time_step, spike_list, Vth, gamma.data, tau_m)
        return np_fn, np_args, np_fn(*np_args), grad_np_args
    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Conv2d, Conv2d).backward(ctx, grad_outputs)


def conv_forward(X, W, stride=1, padding=0):
    # cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = jnp.matmul(W_col, X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = jnp.transpose(out, (3, 0, 1, 2))

    return out
