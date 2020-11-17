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
                jnp.multiply(jnp.subtract(jnp.matmul(input_np, weights_np), v_current), dt * tau_m)+v_current, time_step)

        def grad_fn(grad_outputs, input, s_time_list, time, tau_m, gamma, Vth):
            return (jnp.divide(grad_outputs,time), jnp.matmul(input.T,jnp.divide(grad_outputs,time)))

        np_args = (input.data, weights.data, v_current.data, time_step, dt, tau_m)
        spike = jit(np_fn)(*np_args)

        grad_np_args = (spike, input.data, time_step, tau_m, gamma.data, Vth)

        id = "output"
        return grad_fn, grad_np_args, spike, v_current.data, id

    @staticmethod
    def backward(ctx, grad_outputs):
        super(Output, Output).backward(ctx, grad_outputs)
