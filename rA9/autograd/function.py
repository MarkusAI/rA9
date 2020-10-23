from jax import jit
from .variable import *

def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    return type.__new__(metaclass, 'temporary_class', (), {})

class BackwardFunction(object):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)

class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _is_legacy: True if forward is not defined as a static method.
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """

    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            forward = super_cls.__dict__.get('forward')
            if forward is not None:
                has_static_forward = isinstance(forward, staticmethod) or isinstance(forward, classmethod)
                break

        setattr(cls, '_is_legacy', not has_static_forward)

        # old-style functions
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
    def setup_grad_fn(grad_fn, np_fn, np_args, *args):
        grad_fn.saved_variables = ()
        grad_fn.next_functions = ()
        grad_fn.needs_input_grad = ()
        grad_fn.np_fn = np_fn
        grad_fn.args = args
        grad_fn.np_args = np_args

        for arg in args:
            if isinstance(arg, Variable):
                grad_fn.saved_variables = grad_fn.saved_variables + (arg, )
                if arg.requires_grad:
                    grad_fn.needs_input_grad = grad_fn.needs_input_grad + (True, )
                else:
                    grad_fn.needs_input_grad = grad_fn.needs_input_grad + (False,)

                if arg.grad_fn is not None:
                    grad_fn.next_functions = grad_fn.next_functions + (arg.grad_fn,)
                else:
                    if arg.requires_grad:
                        grad_fn.next_functions = grad_fn.next_functions + (AccumulateGrad(arg), )
            else:
                grad_fn.needs_input_grad = grad_fn.needs_input_grad + (False, )


    @classmethod

    def apply(cls, *args):
        if getattr(cls(), 'id') == 'output':
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()
            np_fn, np_args, output, v_current = cls.forward(grad_fn, *args)
            return Variable(data=output, requires_grad=True,grad_fn =grad_fn), Variable(data=v_current)
        elif getattr(cls(), 'id') == 'LIF':
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()

            np_fn, np_args, output, v_current = cls.forward(grad_fn, *args)

            cls.setup_grad_fn(grad_fn, np_fn, np_args, *args)
            return Variable(data=output, requires_grad=True, gamma=args[4], grad_fn=grad_fn), \
                   Variable(data=v_current), Variable(np_args[0])
        else:
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()
            np_fn, np_args, output = cls.forward(grad_fn, *args)
            cls.setup_grad_fn(grad_fn, np_fn, np_args, *args)
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
