import jax.numpy as jnp
import jax.random as random
from ..networks.module import Module
from jax.ops import index, index_update, index_add


class Input(Module):
    def __init__(self, duration, dt, key=0):
        super().__init__()
        self.dt = dt
        self.duration = duration
        self.key_x = random.PRNGKey(key)

    def forward(self, intensities):
        assert jnp.all(intensities >= 0), "Inputs must be non-negative"
        assert intensities.dtype == jnp.float32 or intensities.dtype == jnp.float64, "Intensities must be of type Float."

        # Get shape and size of data.
        shape, size = jnp.shape(intensities), jnp.size(intensities)
        intensities = intensities.reshape(-1)
        time = self.duration // self.dt

        # Compute firing rates in seconds as function of data intensity,
        # accounting for simulation time step.
        rate_p = jnp.zeros(size)
        non_zero = intensities != 0
        rate = index_update(
            rate_p, index[non_zero], 1 / intensities[non_zero] * (1000 / self.dt))
        del rate_p

        # Create Poisson distribution and sample inter-spike intervals
        # (incrementing by 1 to avoid zero intervals).
        intervals_p = random.poisson(key=self.key_x, lam=rate, shape=(
            time + 1, len(rate))).astype(jnp.float32)
        intervals = index_add(intervals_p, index[:, intensities != 0], (
            intervals_p[:, intensities != 0] == 0).astype(jnp.float32))
        del intervals_p

        # Calculate spike times by cumulatively summing over time dimension.
        times_p = jnp.cumsum(intervals, dtype=float)
        times_p = times_p.reshape((-1, size))
        times = index_update(times_p, times_p >= time + 1, 0).astype(jnp.int32)
        del times_p

        # Create tensor of spikes.
        spike = jnp.zeros((time + 1, size), dtype=bool)
        spikes = index_update(spike, index[times, jnp.arange(size)], 1)
        spikes = spikes[1:]
        return spikes.reshape(time, *shape)


# Implemented from PySNN Poisson Encoding & Bindsnet Poisson encoding
# https://github.com/BasBuller/PySNN
# https://github.com/Hananel-Hazan/bindsnet
