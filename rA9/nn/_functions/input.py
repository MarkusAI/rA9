import numpy as np
import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable


class Input(Function):
    id = "Input"
    @staticmethod
    def forward(ctx, input):
        assert isinstance(input, Variable)

        def np_fn(input_np):
            return jnp.where(input_np>(0.5+np.random.randn(*input_np.shape)), 1, 0)
        id = "Input"
        return np_fn(input.data), id 
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Input, Input).backward(ctx, grad_outputs)
