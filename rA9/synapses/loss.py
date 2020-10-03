

import jax.numpy as jnp
import jax.scipy.signal as signal


# LOSS
class CrossEntropy():

    @staticmethod
    def forward(ctx, input, target, size_average=True):

        def jnp_fn(input, targets, size_average=True):
            probs = jnp.exp(input - jnp.max(input, axis=1, keepdims=True))
            probs /= jnp.sum(probs, axis=1, keepdims=True)
            N = input.shape[0]

            ll = jnp.log(jnp.array(probs[jnp.arange(N), targets]))

            if size_average:
                return -jnp.sum(ll / N)
            else:
                return -jnp.sum(ll)

        jnp_args = (input, target, size_average)
        out = jnp_fn(*jnp_args)
        return out


class MSELoss():

    @staticmethod
    def forward(ctx, input, target, size_average=True):

        def jnp_fn(input_jnp, target_jnp, size_average=True):
            if size_average:
                return jnp.mean((input_jnp - target_jnp) ** 2)
            else:
                return jnp.sum((input_jnp - target_jnp) ** 2)

        jnp_args = (input, target, size_average)
        out = jnp_fn(*jnp_args)
        return out
