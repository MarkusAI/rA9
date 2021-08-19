import jax.numpy as jnp
from .variable import Variable
from jax.ops import index, index_add
from .function import AccumulateGrad
from rA9.nn._functions.img2col import *

gamma_stack = []




def excute(fn, grad_in=None):
    if fn is not None:

        if isinstance(fn, AccumulateGrad):

            if fn.variable.requires_grad and grad_in is not None:

                if fn.variable.grad is None:
                    fn.variable.grad = jnp.zeros(fn.variable.data.shape)
           
              
                if fn.variable.id == None:
                    pass
                else:
                    gamma = gamma_stack.pop()
                    fn.variable.grad = index_add(fn.variable.grad, index[:], gamma * grad_in.T)

            return
        grad_outs, gamma = fn.apply(grad_in)
        if gamma is not None:

            gamma_stack.append(gamma)

        if type(grad_outs) is not tuple:
            grad_outs = (grad_outs,)
        for i, next_func in enumerate(fn.next_functions):
            excute(next_func, grad_outs[i])


def backward(variables):
    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    for variable in variables:
        if variable.grad_fn is not None:
            excute(variable.grad_fn)
