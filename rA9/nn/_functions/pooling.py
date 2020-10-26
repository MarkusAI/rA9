from jax import jit
from .img2col import *
from rA9.autograd import Function
from rA9.autograd import Variable
import jax.numpy as jnp


class Pooling(Function):
    id = "Pooling"

    @staticmethod
    def forward(ctx, input, size, weights, stride=2):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def np_fn(input_np, weight, size, stride=2):
            out, X_col = pool_forward(input_np, weight, size=size, stride=stride)
            return out, X_col

        np_args = (input.data, weights.data, size, stride)
        out, x_col = np_fn(*np_args)
        grad_args = (weights.data, x_col, size, stride)
        id = "Pooling"
        return pool_backward, grad_args, out, id

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Pooling, Pooling).backward(ctx, grad_outputs)


def pool_forward(X, W, size=2, stride=2):
    n, d, h, w = X.shape

    h_out = (h - size) // stride + 1
    w_out = (w - size) // stride + 1


    X_reshaped = X.reshape(n * d, 1, h, w)

    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    max_idx_X = jnp.mean(X_col, axis=0, dtype='int32')
    n_filter, v, h_filter, w_filter = W.shape
    W_col = W.reshape(n_filter, -1)

    out = jnp.matmul(W_col, X_col)

    out = jnp.array(jnp.take(out, max_idx_X))

    out = out.reshape(h_out, w_out, n, d)

    out = jnp.transpose(out, (2, 3, 0, 1))
    out = jnp.array(out, dtype='float32')

    return out, X_col


def pool_backward(X, W, X_col, size=2, stride=2):

    n_filter, v, h_filter, w_filter = W.shape

    dx = jnp.transpose(X, (2, 3, 0, 1))
    dx = jnp.ravel(dx) * (size * size)
    dX = dx
    for i in range((size * size) - 1):
        dX = jnp.append(dX, dx, axis=0)

    dx = dX.reshape(size * size, dx.shape[0])
    W_col = W.reshape(n_filter, -1)

    dout = jnp.matmul(W_col, dx)
    dw = jnp.matmul(dout, X_col.T)
    dw = dw.reshape(W.shape)

    return dw
