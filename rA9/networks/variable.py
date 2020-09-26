import jax.numpy as jnp
from .basic_func import *


class AccumulateGrad():
    def __init__(self, variable):
        self.variable = variable

    def apply(self):
        pass
#클래스 선언

def excute(fn, grad_in=None):
    if fn is not None:
        if isinstance(fn, AccumulateGrad):
            if fn.variable.requires_grad and grad_in is not None:
                if fn.variable.grad is None:
                    fn.variable.grad = jnp.zeros(fn.variable.data.shape)

                fn.variable.grad += grad_in
            return

        grad_outs = fn.apply(grad_in)
        if type(grad_outs) is not tuple:
            grad_outs = (grad_outs,)

        for i, next_func in enumerate(fn.next_functions):
            excute(next_func, grad_outs[i])


def backward(variables):
    variables = (variables,) if isinstance(variables, Variable) else tuple(variables)

    for variable in variables:

        if variable.grad_fn is not None:
            excute(variable.grad_fn)
# 끊기 헷갈림 방지


class Variable(object):

    def __init__(self, data, requires_grad=False, grad_fn=None):

        self.data = data
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None

    def uniform(self, low=None, high=None):
        self.data = jnp.random.uniform(low=low, high=high, size=self.data.shape)

    def get_grad_accumulator(self):
        if self.grad_fn is not None:
            raise RuntimeError("get_grad_accumulator() should be only called on leaf Variables")

        if not self.requires_grad:
            return None

    def backward(self):
        if self.size > 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")

        backward(self)

    def _add(self, other):
        if isinstance(other, Variable):
            return Add.apply(self, other)
        else:
            raise NotImplementedError("")

    def add(self, other):
        return self._add(other)

    def add_(self, other):
        return self._add(other)

    def view(self, *sizes):
        return View.apply(self, sizes)

    def __add__(self, other):
        return self.add(other)

    __radd__ = __add__

    def __iadd__(self, other):
        raise NotImplementedError("")

    _fallthrough_methods = {
        'size',
        'dim'
    }

    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        return object.__getattribute__(self, name)
