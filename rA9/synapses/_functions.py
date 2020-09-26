from ..networks.function import Function
from ..networks.variable import Variable
from ..synapses.img2col import *

import jax.numpy as jnp
import jax.scipy.signal as signal

conv = signal.convolve


# conv2d

class convolution2d(Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None, stride=1, padding=0):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def jnp_fn(input_jnp, weights_jnp, bias=None, stride=1, padding=0):
            out = conv_forward(input_jnp, weights_jnp, bias, stride, padding)

            if bias is None:
                return out
            else:
                return out

        jnp_args = (input.data, weights.data, None if bias is None else bias.data)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(convolution2d, convolution2d).backward(ctx, grad_output)


def conv_forward(X, W, b, stride=1, padding=0):
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
    if b is not None:
        out += b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = jnp.transpose(out, (3, 0, 1, 2))

    return out


# linear
class linear(Function):

    @staticmethod
    def forward(ctx, input, weights, bias=None):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)

        def jnp_fn(input_jnp, weights_jnp, bias):
            out = jnp.matmul(input_jnp, weights_jnp.T)

            if bias is None:
                return out
            else:
                return out + bias

        jnp_args = (input.data, weights.data, None if bias is None else bias.data)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(linear, linear).backward(ctx, grad_output)


# pooling


class Max_pool2d(Function):

    @staticmethod
    def forward(ctx, input, kernel_size):
        assert isinstance(input, Variable)

        def jnp_fn(input_jnp, kernel_size):
            return _pool_forward(input_jnp, kernel_size)

        jnp_args = (input.data, kernel_size)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Max_pool2d, Max_pool2d).backward(ctx, grad_output)


def _pool_forward(X, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    max_idx = jnp.argmax(X_col, axis=0)
    out = jnp.array(X_col[max_idx, range(max_idx.size)])

    out = out.reshape(h_out, w_out, n, d)
    out = jnp.transpose(out, (2, 3, 0, 1))

    return out


class ReLU(Function):

    @staticmethod
    def forward(ctx, input):
        assert isinstance(input, Variable)

        def jnp_fn(input_jnp):
            return input_jnp * (input_jnp > 0)

        jnp_args = (input.data,)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(ReLU, ReLU).backward(ctx, grad_output)


class dropout(Function):

    @staticmethod
    def forward(ctx, input, p=0.5, train=False):
        assert isinstance(input, Variable)

        def jnp_fn(input_jnp, noise):
            return input_jnp * noise

        noise = jnp.random.binomial(1, p, size=input.data.shape)
        if not train:
            noise.fill(1)
        if p == 1:
            noise.fill(0)
        jnp_args = (input.data, noise)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(dropout, dropout).backward(ctx, grad_output)


# LOSS
class CrossEntropy(Function):

    @staticmethod
    def forward(ctx, input, target, size_average=True):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def jnp_fn(input, targets, size_average=True):
            probs = jnp.exp(input - jnp.max(input, axis=1, keepdims=True))
            probs /= jnp.sum(probs, axis=1, keepdims=True)
            N = input.shape[0]

            ll = jnp.log(jnp.array(probs[jnp.arange(N), targets]))

            if size_average:
                return -jnp.sum(ll / N)
            else:
                return -jnp.sum(ll)

        jnp_args = (input.data, target.data, size_average)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(CrossEntropy, CrossEntropy).backward(ctx, grad_output)


class MSELoss(Function):

    @staticmethod
    def forward(ctx, input, target, size_average=True):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def jnp_fn(input_jnp, target_jnp, size_average=True):
            if size_average:
                return jnp.mean((input_jnp - target_jnp) ** 2)
            else:
                return jnp.sum((input_jnp - target_jnp) ** 2)

        jnp_args = (input.data, target.data, size_average)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(MSELoss, MSELoss).backward(ctx, grad_output)
