import rA9
import jax.numpy as jnp
import numpy.random as nprand


class Variable(object):

    def __init__(self, data, id=None, requires_grad=False, grad_fn=None):
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn
        self.id = id

        self.requires_grad = requires_grad

    def grad_fill_zero(self):

        if self.grad is not None:
            self.grad = jnp.zeros(self.grad.shape)

    def normal(self, mean=None, stdv=None):
        self.data = jnp.array(nprand.normal(loc=mean, scale=stdv, size=self.data.shape))

    def get_grad_accumulator(self):
        if self.grad_fn is not None:
            raise RuntimeError("get_grad_accumulator() should be only called on leaf Variables")

        if len(jnp.argwhere(self.gamma == 0)) != 0 and self.requires_grad:
            return jnp.zeros(shape=self.gamma)

    def backward(self):
        if self.size > 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")

        rA9.autograd.backward(self)

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


from ._functions import *
