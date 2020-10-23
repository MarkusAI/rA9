from jax import jit
import jax.numpy as jnp
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable
from jax.ops import index, index_add


class Linear(Function):
    id = "Linear"

    @staticmethod
    def forward(ctx, input, time_step, weights, v_current, gamma, tau_m, Vth, dt):
        assert isinstance(input, Variable)
        assert isinstance(gamma, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)

        # assert isinstance(spike_time, Variable)

        def np_fn(input_np, weights_np, v_current_np, gamma_np, tau_m, Vth, dt):
            # gamma reset problem
            inv_current = jnp.matmul(input_np, weights_np.T)
            if inv_current.shape == v_current.data.shape:
                spike_list, v_current_n = jit(jnp_fn)(x=inv_current, v_current=v_current_np,
                                                      tau_m=tau_m, Vth=Vth, dt=dt)
                return spike_list, v_current_n, index_add(gamma_np.T, index[:], spike_list)


            else:
                spike_list, v_current_n = jit(jnp_fn)(x=inv_current, v_current=v_current_np.T,
                                                  tau_m=tau_m, Vth=Vth, dt=dt)

                return spike_list, v_current_n, index_add(gamma_np.T, index[:], spike_list)

        np_args = (input.data, weights.data, v_current.data, gamma.data, tau_m, Vth, dt)

        spike, v_current_n, gamma_np = np_fn(*np_args)
        gamma.data = gamma_np
        spike_time = jnp.multiply(spike, dt * time_step)
        spike_time = jnp.concatenate((spike, spike_time), axis=1)
        np_grad_args = (weights.data, spike_time, time_step, Vth, gamma.data, tau_m)
        return np_fn, np_grad_args, spike, v_current_n

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Linear, Linear).backward(ctx, grad_outputs)
