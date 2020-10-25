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
            out = pool_forward(input_np, weight, size=size, stride=stride)
            return out

        np_args = (input.data, weights.data, size, stride)
        grad_args = (weights.data, size, stride)
        id = "Pooling"
        return pool_backward, grad_args, np_fn(*np_args), id

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

    X_col = jnp.matmul(W_col, X_col)

    out = jnp.array(jnp.take(X_col, max_idx_X))
    out = out.reshape(h_out, w_out, n, d)

    out = jnp.transpose(out, (2, 3, 0, 1))
    out = jnp.array(out, dtype='float32')
    return out


def pool_backward(X, W, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) // stride + 1
    w_out = (w - size) // stride + 1

    X_col = col2im_indices(X, X.shape, size, size, padding=0, stride=stride)

    max_idx_X = jnp.mean(X_col, axis=0, dtype='int32')
    n_filter, v, h_filter, w_filter = W.shape
    W_col = W.reshape(n_filter, -1)

    X_col = jnp.matmul(W_col, X_col)

    out = jnp.array(jnp.take(X_col, max_idx_X))
    out = out.reshape(h_out, w_out, n, d)

    out = jnp.transpose(out, (2, 3, 0, 1))
    out = jnp.array(out, dtype='float32')
    return out
