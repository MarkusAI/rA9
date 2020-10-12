from jax import jit
import jax.numpy as jnp
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable
from jax.ops import index, index_add


class Linear(Function):
    id = "Linear"
    @staticmethod
    def forward(ctx, input, weights, v_current, gamma,tau_m, Vth, dt,time_step):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)
        assert isinstance(gamma, Variable)

        def np_fn(input_np, weights_np, v_current_np, gamma_np, tau_m, Vth, dt):
            inv_current = jnp.matmul(input_np, weights_np)
            spike_list, v_current_n = jit(jnp_fn)(x=inv_current, v_current=v_current_np,
                                                  tau_m=tau_m, Vth=Vth, dt=dt)
            index_add(gamma_np, index[:], spike_list)

            return spike_list, v_current_n

        np_args = (input.data, weights.data)
        spike_list,v_current_n = np_fn(*np_args)
        grad_np_args =(weights.data,time_step,spike_list,Vth,gamma,tau_m)
        return np_fn, np_args, np_fn(*np_args),grad_np_args


    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Linear, Linear).backward(ctx, grad_outputs)
