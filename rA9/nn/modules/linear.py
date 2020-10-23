from rA9.autograd import Function
from rA9.autograd import Variable
import jax.numpy as np
from jax import jit
from jax import grad

class Linear(Function):

    @staticmethod
    def forward(ctx, input, weight):
        assert isinstance(input, Variable)
        assert isinstance(weight, Variable)

        def np_fn(input_np, weights_np):
            return np.matmul(input_np, weights_np.T)
        np_args = (input.data, weight.data)
        return grad(np_fn), np_args, jit(np_fn)(*np_args), 0, 0

    @staticmethod
    def backward(ctx, grad_output):
        return super(Linear, Linear).backward(ctx, grad_output)
