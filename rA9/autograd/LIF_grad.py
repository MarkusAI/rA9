from jax import jit
from jax import grad
import jax.numpy as jnp


def lif_grad(grad_output, *args):
    def grad(grad_output, weights, time_step, spike_list, Vth, gamma, tau_m):
        return jnp.multiply(
            jnp.matmul(weights, grad_output),
            jnp.multiply(
                (1 / Vth),
                (1 + jnp.multiply(
                    jnp.divide(
                        (1, gamma),
                        jnp.multiply(
                            jnp.exp(
                                jnp.divide(
                                    jnp.subtract(
                                        time_step, spike_list), tau_m)
                            ), (-1 / tau_m)
                        )
                    )
                )
                 )
            )
        )

    return jit(grad)(grad_output, *args)


def loss_grad(input, target, timestep):
    def np_fn(input, target, timestep):
        return (1/2)*jnp.sum((input-target)**2)
    return (grad(np_fn)(input, target))/timestep

