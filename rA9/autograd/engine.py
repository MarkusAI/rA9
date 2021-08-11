import jax.numpy as jnp
from .variable import Variable
from jax.ops import index, index_add
from .function import AccumulateGrad
from rA9.nn._functions.img2col import *
gamma_stack = []


def gammapops(grad_in, n_filter,h_filter):
    gamma = gamma_stack.pop()
    gamma = im2col_indices(gamma,n_filter,h_filter)
    if gamma.shape[1] == grad_in.shape[1]:
        return gamma
    else:
        return gammapops(grad_in, n_filter,h_filter)


def linearpops(n_filter):
    gamma = gamma_stack.pop()
    if len(gamma.shape) == 4:
        gamma = gamma.reshape(gamma.shape[0],-1)

        if gamma.shape[1] == n_filter:
            return gamma
        else:
            return linearpops(n_filter)
    else:
        return gamma


def excute(fn, grad_in=None):
    if fn is not None:

        if isinstance(fn, AccumulateGrad):

            if fn.variable.requires_grad and grad_in is not None:

                if fn.variable.grad is None:
                    fn.variable.grad = jnp.zeros(fn.variable.data.shape)

                grad_in = jnp.where(grad_in == jnp.inf, 0, grad_in)
                grad_in = jnp.nan_to_num(grad_in, copy=False)
                if len(grad_in.shape) != 4:
                    if len(fn.variable.grad.shape) == 4:
                        gamma = gammapops(grad_in, fn.variable.data.shape[2],fn.variable.data.shape[3])

                        grad_in = jnp.matmul(gamma, grad_in.T)
                        grad_in = grad_in.reshape(fn.variable.grad.shape)
                    else:
                        if grad_in.shape != fn.variable.grad.shape:
                            gamma = linearpops(fn.variable.grad.shape[1])
                            grad_in = jnp.transpose(jnp.matmul(gamma.T, grad_in))

                grad_in = jnp.where(grad_in == jnp.inf, 0, grad_in)
                grad_in = jnp.nan_to_num(grad_in, copy=False)
                grad_in = jnp.where(grad_in < 0, 0, grad_in)
                if fn.variable.id == None:
                    pass
                else:
                    fn.variable.grad = index_add(fn.variable.grad, index[:], grad_in)

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
