import jax.numpy as jnp
from jax.ops import index, index_add


class LIF(object):
    # tau => Time Constant; Capacity * Resistance
    # Vmem => Rest Voltage

    def __init__(self, tau_m, Vmem, Vth, dt):
        self.tau_m = tau_m
        self.Vmem = Vmem
        self.Vth = Vth
        self.dt = dt

    def forward(self, x, v_current):
        dV_tau = jnp.divide(jnp.subtract(x, self.Vmem), self.dt)
        dV = jnp.divide(dV_tau, self.tau)
        v_current = index_add(v_current, index[:], dV)
        spike_list = jnp.greater_equal(v_current, self.Vmem).astype(int)
        v_current = jnp.where(v_current >= self.Vth, self.Vmem, v_current * jnp.exp(-1 / self.tau_m))

        return spike_list, v_current

    def backward(self, time, spike_list, weights, e_gradient):
        gamma = spike_list[0]
        t_Tk_divby_tau_m = jnp.divide(jnp.subtract(time, spike_list[1]), self.tau_m)
        f_prime_t = jnp.multiply(jnp.exp(t_Tk_divby_tau_m), (-1 / self.tau_m))
        aLIFnet = jnp.multiply(1 / self.Vth, (1 + jnp.multiply(jnp.divide(1, gamma), f_prime_t)))
        d_w = jnp.matmul(weights, e_gradient)

        return jnp.multiply(d_w, aLIFnet)
