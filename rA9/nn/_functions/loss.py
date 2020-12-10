from jax import jit
import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable


class Spikeloss(Function):
    id = "Spikeloss"

    @staticmethod
    def forward(ctx, input, target, time_step):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input_np, target_np, time_step):
            return jnp.sum((input_np - jnp.eye(input_np.shape[1])[target_np]) ** 2) / 2
        
        def grad_fn(input_np, target_np, time_step):
            return input_np - jnp.eye(input_np.shape[1])[target_np]

        np_args = (input.data, target.data, time_step)
        id = "Spikeloss"

        return grad_fn, np_args, jit(np_fn)(*np_args), id

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Spikeloss, Spikeloss).backward(ctx, grad_outputs)
