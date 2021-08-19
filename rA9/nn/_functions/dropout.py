import numpy as np
from rA9.autograd import Function
from rA9.autograd import Variable

class Dropout(Function):
    id = "Dropout"
    @staticmethod
    def forward(ctx, input, p=0.2, train=False):
        assert isinstance(input, Variable)
        mask = np.where(np.random.rand(*input.data.shape) > p, 1, 0)
        def np_fn(input, mask, p, train):
            if train:
                return input*mask
            else:
                return input*(1.0 - p)
        
        def grad_fn(grad_outputs, mask):
            return grad_outputs*mask
        
        np_args = (input.data, mask, p, train)
        grad_args = (mask)

        id = "Dropout"
        return grad_fn, grad_args, np_fn(*np_args),id

    @staticmethod
    def backward(ctx, grad_output):
        return super(Dropout, Dropout).backward(ctx, grad_output)
