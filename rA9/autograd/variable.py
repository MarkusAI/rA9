import rA9
import jax.numpy as jnp
import jax.random as random


class Variable(object):

    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad

    def grad_fill_zero(self):

        if self.grad is not None:
            self.grad = jnp.zeros(self.grad.shape)

    def uniform(self, low=None, high=None):
        key = random.PRNGKey(0)
        self.data = random.uniform(minval=low, maxval=high, shape=self.data.shape, key=key)

    def get_grad_accumulator(self):
        if self.grad_fn is not None:
            raise RuntimeError("get_grad_accumulator() should be only called on leaf Variables")

        if not self.requires_grad:
            return None


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
