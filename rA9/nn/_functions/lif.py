import jax.numpy as jnp
from jax.ops import index, index_add


def jnp_fn(x, v_current, tau_m, Vth, dt):
    return jnp.greater_equal(
        index_add(v_current, index[:], jnp.divide(
            (
                jnp.multiply(jnp.subtract(x, v_current), dt)
            ), tau_m

        )
                  )
        , Vth).astype('int32'), jnp.where(v_current >= Vth, 0,
              v_current * jnp.exp(-1 / tau_m))

