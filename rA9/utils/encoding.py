import jax.numpy as jnp
import numpy.random as random
from jax.ops import index, index_update

def poisson_encoding(intensities, duration, dt):
    assert jnp.all(intensities >= 0), "Inputs must be non-negative"
    assert intensities.dtype == jnp.float32 or intensities.dtype == jnp.float64, "Intensities must be of type Float."

    # Get shape and size of data.
    shape, size = jnp.shape(intensities), jnp.size(intensities)
    intensities = intensities.reshape(-1)
    time = duration // dt

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate_p = jnp.zeros(size)
    non_zero = intensities != 0
    rate = index_update(rate_p, index[non_zero], 1 / intensities[non_zero] * (1000 / dt))

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    #have to adapt this part to only jax, not numpy.
    intervals = random.poisson(lam=rate, size=(time + 1, len(rate))).astype(jnp.float32)
    intervals[:, intensities != 0] += (intervals[:, intensities != 0] == 0).astype(jnp.float32)

    # Calculate spike times by cumulatively summing over time dimension.
    times_p = jnp.cumsum(intervals, dtype=float)
    times_p= times_p.reshape((-1, size))
    times = index_update(times_p, times_p >= time+1, 0).astype(jnp.int32)

    # Create tensor of spikes.
    spike = jnp.zeros((time+1, size), dtype=bool)
    spikes = index_update(spike, index[times, jnp.arange(size)], 1)
    spikes = spikes[1:]
    spikes = jnp.moveaxis(spikes, 1, 0)
    return spikes.reshape(*shape, time)


class PoissonEncoder:

    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt

    def __call__(self, intensities):
        return poisson_encoding(intensities, self.duration, self.dt)


# Implemented from PySNN Poisson Encoding & Bindsnet Poisson encoding
# https://github.com/BasBuller/PySNN
# https://github.com/Hananel-Hazan/bindsnet
