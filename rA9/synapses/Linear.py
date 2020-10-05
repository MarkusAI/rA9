import math
from functools import wraps

import jax
import jax.numpy as jnp
from jax import jit
from jax import linear_util as lu
from jax import vjp
from jax.api import _check_scalar, argnums_partial

from ..networks.module import Module
from ..synapses.img2col import *

# 함수 정의


def elementwise_grad(function, x, initial_gradient=None):
    gradient_function = grad(function, initial_gradient, x)
    return gradient_function


def grad(fun, initial_grad=None, argnums=0):
    value_and_grad_f = value_and_grad(fun, initial_grad, argnums)

    docstr = ("Gradient of {fun} with respect to positional argument(s) "
              "{argnums}. Takes the same arguments as {fun} but returns the "
              "gradient, which has the same shape as the arguments at "
              "positions {argnums}.")

    @wraps(fun, docstr=docstr, argnums=argnums)
    def grad_f(*args, **kwargs):
        ans, g = value_and_grad_f(*args, **kwargs)
        return g

    return grad_f


def value_and_grad(fun, initial_grad=None, argnums=0):
    docstr = ("Value and gradient of {fun} with respect to positional "
              "argument(s) {argnums}. Takes the same arguments as {fun} but "
              "returns a two-element tuple where the first element is the value "
              "of {fun} and the second element is the gradient, which has the "
              "same shape as the arguments at positions {argnums}.")

    @wraps(fun, docstr=docstr, argnums=argnums)
    def value_and_grad_f(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args)
        ans, vjp_py = vjp(f_partial, *dyn_args)

        g = vjp_py(jnp.ones((), jnp.result_type(ans))
                   if initial_grad is None else initial_grad)
        g = g[0] if isinstance(argnums, int) else g
        return (ans, g)

    return value_and_grad_f


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = jnp.zeros((out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = 1. / math.sqrt(size[1])
        self.weight = jax.random.uniform(
            minval=-stdv, maxval=stdv, shape=self.weight, key=keyW)
        if self.bias is not None:
            self.bias = jax.random.uniform(
                minval=-stdv, maxval=stdv, shape=self.bias, key=keyB)

    def forward(self, input):
        def jnp_fn(input_jnp, weights_jnp):
            out = jnp.matmul(input_jnp, weights_jnp.T)  # T

            return out

        jnp_args = (input, self.weight,
                    None if self.bias is None else self.bias)
        out = jnp_fn(*jnp_args)
        return out

    def backward(self, grad_outputs):
        jnp_fn = jnp_fn
        jnp_args = self.jnp_args
        indexes = [index for index, need_grad in enumerate(
            self.needs_input_grad) if need_grad]

        jnp_grad_fn = elementwise_grad(jnp_fn, indexes, grad_outputs)
        grads = jnp_grad_fn(*jnp_args)
        return grads
