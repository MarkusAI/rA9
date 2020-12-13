from jax import jit
import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable


class Linear(Function):
    id = "Linear"

    @staticmethod
    def forward(ctx, input, weight):
        assert isinstance(input, Variable)
        assert isinstance(weight, Variable)

        def np_fn(input_np, weights_np):
            out = jnp.matmul(input_np, weights_np)
            return out
        np_args = (input.data, weight.data.T)
        id = "Linear"
        return np_fn, np_args, jit(np_fn)(*np_args), id


    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Linear, Linear).backward(ctx, grad_outputs)
