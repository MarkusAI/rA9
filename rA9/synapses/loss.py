import jax.numpy as jnp


# LOSS
class Spike_LOSS():
    @staticmethod
    def forward(ctx,output,label):
        out = output-label
        return out