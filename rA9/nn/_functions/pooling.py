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
        out = np_fn(*np_args)
        grad_args = (weights.data, size, stride)
        id = "Pooling"
        return pool_backward, grad_args, out, id

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Pooling, Pooling).backward(ctx, grad_outputs)


def pool_forward(X, W, size=2, stride=2):
    n, d, h, w = X.shape

    h_out = (h - size) // stride + 1
    w_out = (w - size) // stride + 1

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)

    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    n_filter, v, h_filter, w_filter = W.shape
    W_col = W.reshape(n_filter, -1)

    out = jnp.matmul(W_col, X_col)
    out = jnp.mean(out,axis=0)
    out = out.reshape(h_out, w_out, n, d)
    out = jnp.transpose(out, (2, 3, 0, 1))
    out = jnp.array(out, dtype='float32')

    return out


def pool_backward(X,v_currnet, W, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - 1) * stride + size
    w_out = (w - 1) * stride + size
    h_out, w_out = int(h_out), int(w_out)

    dx = jnp.transpose(X, (2, 3, 0, 1))

    dx = jnp.ravel(dx) / (size * size)
    dX = dx
    for i in range((size * size) - 1):
        dX = jnp.append(dX, dx, axis=0)
    dx = jnp.reshape(dX, (size * size, -1))
    change_size = (n * d, 1, h_out, w_out)
    dx = col2im_indices(dx, change_size, size, size, padding=0, stride=stride)
    dx = jnp.reshape(dx,(n,d,h_out,w_out))
    return dx
