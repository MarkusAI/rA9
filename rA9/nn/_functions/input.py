from jax import jit
import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable


class Input(Function):
    id = "LIF"

    @staticmethod
    def forward(ctx, input, v_current, tau_m, Vth, dt, s_time_list, time, gamma):
        assert isinstance(input, Variable)

        def np_fn(input_np, v_current, gamma, tau_m, Vth, dt):
            spike = jnp.greater_equal(input_np + v_current, Vth).astype('float32')
            v_current = input_np + v_current - spike

            return spike, jnp.multiply(jnp.exp(-1 / tau_m), v_current), gamma + spike.astype('int32')

        def grad_fn(grad_outputs, s_time_list, time, tau_m, gamma, Vth):
            return jnp.multiply(grad_outputs, 1 / Vth * (
                    1 + jnp.multiply(1 / gamma, jnp.sum(jnp.multiply(-1 / tau_m, jnp.exp((-1 / tau_m)*(time - s_time_list)))))))

        np_args = (input.data, v_current.data, gamma.data, tau_m, Vth, dt)
        spike, v_current, gamma = jit(np_fn)(*np_args)

        spike_time = spike * time
        s_time_list = jnp.concatenate((spike_time, s_time_list.data))
        grad_np_args = (s_time_list, time, tau_m, gamma, Vth)

        id = "LIF"

        return grad_fn, grad_np_args, spike, v_current, gamma, s_time_list, id

    @staticmethod
    def backward(ctx, grad_output):
        return super(Input, Input).backward(ctx, grad_output)