import jax.numpy as jnp


# LOSS
class Spike_LOSS():
    @staticmethod
    def forward(ctx,output,label,timestep):
        out = 1/2*jnp.sum((output-label)**2)
        e_grad= (output-label)/timestep
        return out ,e_grad
