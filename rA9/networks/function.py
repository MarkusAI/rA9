from .variable import *

import jax.numpy as jnp

from ..networks.autograd_style import make_vjp
from autograd.extend import primitive, defvjp_argnum, vspace
from autograd.wrap_util import unary_to_nary

#함수 정의
@unary_to_nary
def elementwise_grad(fun, x, initial_grad=None):
    vjp, ans = make_vjp(fun, x)
    if vspace(ans).iscomplex:
        raise TypeError("Elementwise_grad only applies to real-output functions.")
    return vjp(vspace(ans).ones() if initial_grad is None else initial_grad)


# 헷갈림 방지선

def with_metaclass(meta, *bases):
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

    return type.__new__(metaclass, 'temporary_class', (), {})


class BackwardFunction(object):
    _is_legacy = False

    def apply(self, *args):
        return backward(self,*args)


class FunctionMeta(type):
    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            forward = super_cls.__dict__.get('forward')
            if forward is not None:
                has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
                break

        setattr(cls, '_is_legacy', not has_static_forward)

        if not has_static_forward:
            return super(FunctionMeta, cls).__init__(name, bases, attrs)

        backward_fn = type(name + 'Backward', (BackwardFunction,), {'_forward_cls': cls})
        setattr(cls, '_backward_cls', backward_fn)

        return super(FunctionMeta, cls).__init__(name, bases, attrs)


class AccumulateGrad():
    def __init__(self, variable):
        self.variable = variable

    def apply(self):
        pass


class Function(with_metaclass(FunctionMeta)):

    @staticmethod
    def setup_grad_fn(grad_fn, jnp_fn, jnp_args, *args):
        grad_fn.saved_variables = ()
        grad_fn.next_functions = ()
        grad_fn.needs_input_grad = ()
        grad_fn.np_fn = jnp_fn
        grad_fn.args = args
        grad_fn.np_args = jnp_args

        for arg in args:
            if isinstance(arg, Variable):
                grad_fn.saved_variables = grad_fn.saved_variables + (arg,)
                if arg.requires_grad:
                    grad_fn.needs_input_grad = grad_fn.needs_input_grad + (True,)
                else:
                    grad_fn.needs_input_grad = grad_fn.needs_input_grad + (False,)

                if arg.grad_fn is not None:
                    grad_fn.next_functions = grad_fn.next_functions + (arg.grad_fn,)
                else:
                    if arg.requires_grad:
                        grad_fn.next_functions = grad_fn.next_functions + (AccumulateGrad(arg),)
            else:
                grad_fn.needs_input_grad = grad_fn.needs_input_grad + (False,)

    @classmethod
    def apply(ctx, *args):

        backward_cls = ctx()._backward_cls
        grad_fn = backward_cls()
        np_fn, np_args, output = ctx.forward(grad_fn, *args)

        ctx.setup_grad_fn(grad_fn, np_fn, np_args, *args)

        out_val = Variable(output, requires_grad=True, grad_fn=grad_fn)

        return out_val

    @staticmethod
    def forward(*args, **kwargs):

        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_outputs):

        np_fn = ctx.np_fn
        np_args = ctx.np_args
        indexes = [index for index, need_grad in enumerate(ctx.needs_input_grad) if need_grad]

        np_grad_fn = elementwise_grad(np_fn, indexes, initial_grad=grad_outputs)
        grads = np_grad_fn(*np_args)
        return grads
