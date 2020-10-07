import jax.numpy as jnp
from ..networks.module import Module
from jax.ops import index, index_add


class Output(Module):
    def __init__(self, tau_m, dt):
        super().__init__()
        self.tau_m = tau_m
        self.dt = dt

    def forward(self, x, v_current, time_step):
        dV_tau = jnp.multiply(jnp.subtract(x, v_current), self.dt)
        dV = jnp.divide(dV_tau, self.tau_m)
        v_current = index_add(v_current, index[:], dV)
        return jnp.divide(v_current * jnp.exp(-1 / self.tau_m), time_step*self.dt)
