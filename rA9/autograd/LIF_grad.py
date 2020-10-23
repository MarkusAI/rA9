from jax import jit
from jax import grad
import jax.numpy as jnp


def lif_grad(grad_output, *args):
    def grad(grad_output, weights, spike_list, time_step, Vth, gamma, tau_m):
        return jnp.multiply(
            jnp.matmul(weights, grad_output),
            jnp.multiply(
                (1 / Vth),
                (1 + jnp.multiply(
                    jnp.divide((1, gamma), jnp.multiply(jnp.exp(jnp.divide(jnp.subtract(time_step, spike_list), tau_m)),
                                                        (-1 / tau_m)))
                )
                 )
            )
        )

    return jit(grad)(grad_output, *args)


'''
def loss_grad(input, target, timestep):
    def np_fn(input, target):
        return (1 / 2) * jnp.sum((jnp.argmax(input) - target.T) ** 2)

    print((jnp.argmax(input) - target.T))
    return (grad(np_fn)(input, target)) / timestep'''

import jax.numpy as np
from jax import jit, wraps, lu
from jax import vjp
import numpy as onp
from jax.api import _argnums_partial, _check_scalar


def elementwise_grad(fun, x, initial_grad=None):
    grad_fun = grad(fun, initial_grad, x)
    return grad_fun


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
        f_partial, dyn_args = _argnums_partial(f, argnums, args)
        ans, vjp_py = vjp(f_partial, *dyn_args)

        g = vjp_py(jnp.ones((), jnp.result_type(ans)) if initial_grad is None else initial_grad)
        g = g[0] if isinstance(argnums, int) else g
        return (ans, g)

    return value_and_grad_f
