import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable
from jax import jit


class Output(Function):
    id = "output"

    @staticmethod
    def forward(ctx, input, weights, v_current, tau_m, dt, time_step, Vth, gamma):
        assert isinstance(input, Variable)
        assert isinstance(v_current, Variable)
        assert isinstance(weights, Variable)

        def np_fn(input_np, weights_np, v_current, time_step, dt, tau_m):
            return jnp.divide(
                jnp.multiply(jnp.subtract(jnp.matmul(input_np, weights_np), v_current), dt * tau_m), time_step)

        def grad_fn(grad_outputs, s_time_list, time, tau_m, gamma, Vth):
            out = jnp.multiply(grad_outputs,
                               (1 / Vth * (1 + jnp.multiply(1 / gamma, jnp.sum(
                                   jnp.multiply(-1 / tau_m, jnp.exp(time - s_time_list)))))))
            out = jnp.where(out == jnp.inf, 0, out)
            out = jnp.nan_to_num(out)
            return out

        np_args = (input.data, weights.data, v_current.data, time_step, dt, tau_m)
        spike = jit(np_fn)(*np_args)

        grad_np_args = (spike, time_step, tau_m, gamma.data, Vth)

        id = "output"
        return grad_fn, grad_np_args, spike, v_current.data, id

    @staticmethod
    def backward(ctx, grad_outputs):

        super(Output, Output).backward(ctx, grad_outputs)
