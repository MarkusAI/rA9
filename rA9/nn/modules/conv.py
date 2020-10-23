from rA9.autograd import Function
from rA9.autograd import Variable
from jax import numpy as np
from .img2col import *
from jax import grad


class Conv2d(Function):

    @staticmethod
    def forward(ctx, input, weight, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(weight, Variable)

        def np_fn(input_np, weights_np, stride=1, padding=0):
            return conv_forward(input_np, weights_np, stride, padding)

        np_args = (input.data, weight.data)
        return grad(np_fn), np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Conv2d, Conv2d).backward(ctx, grad_output)


def conv_forward(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = np.matmul(W_col, X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = np.transpose(out, (3, 0, 1, 2))

    return out


