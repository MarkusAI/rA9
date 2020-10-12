import jax.numpy as jnp
from jax import jit


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

    if len(jnp.argwhere(args[4] == 0)) != 0:
        return None
    else:
        print(args[4])
        return jit(grad)(grad_output, *args)


def loss_grad(input,target,timestep):
    def lossgrad(input,target,timestep):
        return (input-target)/timestep
    return jit(lossgrad)(input,target,timestep)