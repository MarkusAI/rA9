from jax import jit
import jax.numpy as jnp

from rA9.autograd import Function
from rA9.autograd import Variable

class Dropout(Function):
    id = "Dropout"
    @staticmethod
    def forward(ctx, input, p=0.5, train=False):
        assert isinstance(input, Variable)
        noise = jax.random.bernoulli(1, p, shape=input.data.shape)
        if not train:
            noise = jnp.ones(input.data.shape)
        if p == 1:
            noise = jnp.zeros(input.data.shape)
        def np_fn(input_np, noise):
            return input_np * noise
        np_args = (input.data, noise)
        id = "Dropout"
        return np_fn, np_args, jit(np_fn)(*np_args),id

    @staticmethod
    def backward(ctx, grad_output):
        return super(Dropout, Dropout).backward(ctx, grad_output)


