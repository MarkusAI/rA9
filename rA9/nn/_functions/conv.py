from jax import jit
from .img2col import *
from rA9.autograd import Function
from rA9.autograd import Variable


class Conv2d(Function):
    id = "Conv2d"

    @staticmethod
    def forward(ctx, input, weights, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def np_fn(input_np, weights_np, stride=1, padding=0):
            out, X_col = conv_forward(input_np, weights_np, stride, padding)

            return out, X_col

        np_args = (input.data, weights.data, stride, padding)
        out, X_col = np_fn(*np_args)
        grad_args = (weights.data, X_col, stride, padding)
        id = "Conv2d"
        return conv_backward, grad_args, out, id

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Conv2d, Conv2d).backward(ctx, grad_outputs)


def conv_forward(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.shape

    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) // stride + 1
    w_out = (w_x - w_filter + 2 * padding) // stride + 1

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)

    W_col = W.reshape(n_filters, -1)

    out = jnp.matmul(W_col, X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = jnp.transpose(out, (3, 0, 1, 2))

    return out, X_col


def conv_backward(X, W, X_col, stride=1, padding=0):
    n_filter, v, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - 1) * stride + h_filter - 2 * padding
    w_out = (w_x - 1) * stride + w_filter - 2 * padding

    h_out, w_out = int(h_out), int(w_out)

    dout_reshaped = jnp.transpose(X, (1, 2, 3, 0)).reshape(n_filter, -1)

    W_col = W.reshape(n_filter, -1)
    dx_col = jnp.matmul(W_col.T, dout_reshaped)
    dW = jnp.matmul(dout_reshaped , X_col.T)
    dW = dW.reshape(W.shape)

    newshape = (n_x, v, h_out, w_out)

    dx = col2im_indices(dx_col, newshape, h_filter, w_filter, padding=padding, stride=stride)
    grads =(dx,dW)
    return grads
