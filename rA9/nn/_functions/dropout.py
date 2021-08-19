from jax import jit
from rA9.autograd import Function
from rA9.autograd import Variable

class Dropout(Function):
    id = "Dropout"
    @staticmethod
    def forward(ctx, input, mask, p=0.2, train=False):
        assert isinstance(input, Variable)

        def np_fn(input, mask, p, train):
            if train:
                return input*mask
            else:
                return input*(1.0 - p)
        
        def grad_fn(grad_outputs, mask):
            return grad_outputs*mask
        
        np_args = (input, mask, p, train)
        grad_args = (mask)

        id = "Dropout"
        return grad_fn, grad_args, jit(np_fn)(*np_args),id

    @staticmethod
    def backward(ctx, grad_output):
        return super(Dropout, Dropout).backward(ctx, grad_output)
