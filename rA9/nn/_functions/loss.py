import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable


class Spikeloss(Function):
    self.id = "Spikeloss"

    @staticmethod
    def forward(ctx, input, target, time_step):
        assert isinstance(input, Variable)
        assert isinstance(target, Variable)

        def np_fn(input_np, target_np, time_step):
            input_np = jnp.argmax(input_np)
            return jnp.sum((input_np - target_np) ** 2) / 2

        # target.data -> jnp.array is none..

        np_args = (input.data, target.data, time_step)
        return np_fn, np_args, np_fn(*np_args)

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Spikeloss, Spikeloss).backward(ctx, grad_outputs)
