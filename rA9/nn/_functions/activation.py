from jax import jit
import jax.numpy as np
from rA9.autograd import Function
from rA9.autograd import Variable


class LIF(Function):
    id = "LIF"

    @staticmethod
    def forward(ctx, input, v_current, tau_m, Vth, dt, s_time_list, time, gamma):
        assert isinstance(input, Variable)

        def np_fn(input_np, v_current,gamma, tau_m, Vth, dt):
            v_current = ((input_np-v_current)/tau_m)*dt
            spike = np.greater_equal(v_current + np.multiply(np.divide(np.subtract(input_np, v_current), tau_m), dt),
                                     Vth).astype('float32')

            gamma += np.where(spike>=Vth, 1, 0)
            return spike, v_current, gamma


        def grad_fn(grad_outputs, s_time_list, time, tau_m, gamma, Vth):
            return np.multiply(grad_outputs, (1 / Vth * (
                        1 + np.multiply(1 / gamma, np.sum(np.multiply(-1 / tau_m, np.exp(time - s_time_list)))))))

        np_args = (input.data, v_current.data, gamma.data, tau_m, Vth, dt)
        spike, v_current,gamma = jit(np_fn)(*np_args)

        spike_time = spike * time
        s_time_list = np.concatenate((spike_time, s_time_list.data))
        grad_np_args = (s_time_list, time, tau_m, gamma, Vth)
        id = "LIF"

        return grad_fn, grad_np_args, spike, v_current, id

    @staticmethod
    def backward(ctx, grad_output):
        return super(LIF, LIF).backward(ctx, grad_output)
