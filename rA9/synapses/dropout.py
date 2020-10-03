from rA9.synapses.img2col import *
from rA9.networks.module import Module
from jax import vjp
from jax import linear_util as lu
import jax
from functools import wraps
from jax.api import argnums_partial



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

class Dropout(Module):

    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        def jnp_fn(input_jnp, noise):
            return input_jnp * noise

        noise = jnp.random.binomial(1, self.p, size=input.shape)
        if not train:
            noise.fill(1)
        if p == 1:
            noise.fill(0)
        self.jnp_args = (input, noise)
        out= jnp_fn(*self.jnp_args)
        return out

    def backward(self, grad_ouputs):
        jnp_fn = jnp_fn
        jnp_args = self.jnp_args
        indexes = [index for index, need_grad in enumerate(self.needs_input_grad) if need_grad]

        jnp_grad_fn = elementwise_grad(jnp_fn, indexes, grad_outputs)
        grads = jnp_grad_fn(*jnp_args)
        return grads


