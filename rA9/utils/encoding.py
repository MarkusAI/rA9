import jax.numpy as jnp
import jax.random as random
from jax.ops import index, index_update, index_add


class UniformEncoder(object):
    def __init__(self, duration, mode=1, key=0):
        super().__init__()
        self.mode = mode
        self.duration = duration
        self.key = random.PRNGKey(key)

    def Encoding(self, intensities):
        rnum = random.uniform(key=self.key, shape=(self.duration, *intensities.shape))
        uin = (jnp.abs(intensities)/self.mode > rnum).astype('float32')
        return jnp.multiply(uin, jnp.sign(intensities))



class PoissonEncoder(object):
    def __init__(self, duration, dt=1, key=0):
        super().__init__()
        self.dt = dt
        self.duration = duration
        self.key_x = random.PRNGKey(key)

    def Encoding(self, intensities):
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

        rate = index_update(rate_p, index[non_zero], 1 / intensities[non_zero] * (1000 / self.dt))
        del rate_p

        # Create Poisson distribution and sample inter-spike intervals
        # (incrementing by 1 to avoid zero intervals).
        intervals_p = random.poisson(key=self.key_x, lam=rate, shape=(time, len(rate))).astype(jnp.float32)

        intervals = index_add(intervals_p, index[:, intensities != 0],
                              (intervals_p[:, intensities != 0] == 0).astype(jnp.float32))

        del intervals_p

        # Calculate spike times by cumulatively summing over time dimension.

        times_p = jnp.cumsum(intervals, dtype='float32', axis=0)
        times = index_update(times_p, times_p >= time + 1, 0).astype(bool)

        del times_p

        spikes_p = jnp.zeros(shape=(time + 1, size))
        spikes = index_update(spikes_p, index[times], 1)
        spikes = spikes[1:]
        spikes = jnp.transpose(spikes, (1, 0)).astype(jnp.float32)
        return spikes.reshape(time, *shape)
