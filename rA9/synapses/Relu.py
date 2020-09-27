import jax.numpy as jnp
from ..networks.module import Module

from jax import vjp
from jax import jit, wraps, lu

from jax.api import _argnums_partial, _check_scalar


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


class Relu(Module):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, input):
        def jnp_fn(input_jnp):
            return input_jnp * (input_jnp > 0)

        jnp_args = (input)
        out = jnp_fn(*jnp_args)
        return out

    def backward(self, grad_ouputs):
        jnp_fn = jnp_fn
        jnp_args = self.jnp_args
        indexes = [index for index, need_grad in enumerate(self.needs_input_grad) if need_grad]

        jnp_grad_fn = elementwise_grad(jnp_fn, indexes, grad_outputs)
        grads = jnp_grad_fn(*jnp_args)
        return grads
