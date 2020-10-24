from rA9.autograd import Function
from rA9.autograd import Variable
import jax.numpy as jnp


class Dropout(Function):
    id = "Dropout"
    @staticmethod
    def forward(ctx, input, p=0.5, train=False):
        assert isinstance(input, Variable)

        def np_fn(input_np, noise):
            return input_np * noise

        noise = jnp.random.binomial(1, p, size=input.data.shape)
        if not train:
            noise.fill(1)
        if p == 1:
            noise.fill(0)
        np_args = (input.data, noise)
        id = "Dropout"
        return np_fn, np_args, np_fn(*np_args),id

    @staticmethod
    def backward(ctx, grad_output):
        return super(Dropout, Dropout).backward(ctx, grad_output)
