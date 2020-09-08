import jax.numpy as jnp
import jax 

def poisson_encoding(intensities, duration, dt):
    assert (intensities >= 0).all(), "Inputs must be non-negative."
    assert intensities.dtype == jnp.float, "Intensities must be of type Float."

    # Get shape and size of data.
    shape, size = intensities.shape, intensities.numel()
    intensities = intensities.view(-1)
    time = int(duration / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = jnp.zeros(size)
    non_zero = intensities != 0
    rate[non_zero] = 1 / intensities[non_zero] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = jax.random.poisson(lam=rate)
    intervals = dist.sample(sample_shape=jnp.shape([time + 1]))
    intervals[:, intensities != 0] += (intervals[:, intensities != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = jnp.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = jnp.zeros(time + 1, size).bool()
    spikes[times, jnp.arange(size)] = 1
    spikes = spikes[1:]
    spikes = jnp.moveaxis(spikes,1, 0)

    return spikes.view(*shape, time)


class PoissonEncoder:

    def __init__(self, duration, dt):
        self.duration = duration
        self.dt = dt

    def __call__(self, intensities):
        return poisson_encoding(intensities, self.duration, self.dt)


#Implement the code from PySNN (https://github.com/BasBuller/PySNN/blob/master/pysnn/encoding.py) to JAX(https://github.com/google/jax)
