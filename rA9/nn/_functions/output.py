from jax import jit
import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable


class Output(Function):

    @staticmethod
    def forward(ctx, input, weights, v_current, tau_m, dt, time_step):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)
        assert isinstance(tau_m, Variable)
        assert isinstance(dt, Variable)
        assert isinstance(time_step, Variable)

        Vmem = jnp.divide(
            (
                jnp.multiply(
                    jnp.subtract(
                        jnp.matmul(
                            input.data, weights.data
                        ),
                        v_current),
                    dt)
            ),
            tau_m
        )
        return Vmem / (dt * time_step), dt*time_step

    @staticmethod
    def backward(ctx, grad_outputs):
        super(Output, Output).backward(ctx, grad_outputs)
