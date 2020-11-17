import jax.numpy as jnp
from .variable import Variable
from jax.ops import index, index_add
from .function import AccumulateGrad


def excute(fn, grad_in=None,spike=None):
    if fn is not None:

        if isinstance(fn, AccumulateGrad):

            if fn.variable.requires_grad and grad_in is not None:

                if fn.variable.grad is None:
                    fn.variable.grad = jnp.zeros(fn.variable.data.shape)

                fn.variable.grad = index_add(fn.variable.grad, index[:], grad_in)

            return

        grad_outs,spike = fn.apply(grad_in,spike)

        if type(grad_outs) is not tuple:
            grad_outs = (grad_outs,)

        for i, next_func in enumerate(fn.next_functions):

            excute(next_func, grad_outs[i],spike)


def backward(variables):
    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    for variable in variables:
        if variable.grad_fn is not None:
            excute(variable.grad_fn)
