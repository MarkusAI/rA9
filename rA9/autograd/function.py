from .variable import *
from .LIF_grad import *


def with_metaclass(meta, *bases):
    class metaclass(meta):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

    return type.__new__(metaclass, 'temporary_class', (), {})


class BackwardFunction(object):
    _is_legacy = False

    def apply(self, *args):
        return self._forward_cls.backward(self, *args)


class FunctionMeta(type):

    def __init__(cls, name, bases, attrs):
        FunctionMeta.grad_fn = None
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


class AccumulateGrad:
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
    def apply(cls, *args):

        if getattr(cls(), 'id') == 'Spikeloss':
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()
            np_fn, np_args, output = cls.forward(grad_fn, *args)
            cls.setup_grad_fn(grad_fn, np_fn, np_args, *args)

            return Variable(data=output, requires_grad=True, grad_fn=grad_fn)
        elif getattr(cls(), 'id') == 'View':
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()
            np_fn, np_args, output = cls.forward(grad_fn, *args)
            cls.setup_grad_fn(grad_fn, np_fn, np_args, *args)
            return Variable(data=output, requires_grad=True, grad_fn=grad_fn)
        elif getattr(cls(), 'id') == 'add':
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()
            np_fn, np_args, output = cls.forward(grad_fn, *args)
            cls.setup_grad_fn(grad_fn, np_fn, np_args, *args)
            return Variable(data=output, requires_grad=True, grad_fn=grad_fn)
        
        elif getattr(cls(), 'id') == 'output':
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()
            np_fn, np_args, output = cls.forward(grad_fn, *args)
            return Variable(data=output, requires_grad=False)
        else:
            backward_cls = cls()._backward_cls
            grad_fn = backward_cls()

            np_fn, np_args, output, v_current = cls.forward(grad_fn, *args)

            cls.setup_grad_fn(grad_fn, np_fn, np_args, *args)
            return Variable(data=output, requires_grad=True, gamma=args[4], grad_fn=grad_fn), Variable(v_current)

    @staticmethod
    def forward(*args, **kwargs):

        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_outputs):

        if len(ctx.np_args) == 3:
            grads = ctx.np_fn(*ctx.np_args)

            return grads
        else:
            if len(ctx.np_args) == 2:
                grads = grad_outputs
            else:
                np_args = ctx.np_args
                grads = lif_grad(grad_outputs, *np_args)
            return grads
