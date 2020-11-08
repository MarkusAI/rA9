from collections import defaultdict

import rA9
import jax.numpy as jnp
from rA9.autograd import Variable

required = object()


class Optimizer(object):

    def __init__(self, params, defaults):

        self.state = defaultdict(dict)
        self.param_groups = list(params)
        if len(self.param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(self.param_groups[0], dict):
            self.param_groups = [{'params': self.param_groups}]
        for group in self.param_groups:
            for param in group['params']:
                if not isinstance(param, Variable):
                    raise TypeError("optimizer can only optimize Variables, "
                                    "but one of the params is " + rA9.typename(param))
                if not param.requires_grad:
                    raise ValueError("optimizing a parameter that doesn't "
                                     "require gradients")

        if isinstance(params, Variable):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Variables or dicts, but got " +
                            rA9.typename(params))

        if not isinstance(self.param_groups[0], dict):
            self.param_groups = [{'params': self.param_groups}]

        param_set = set()
        for group in self.param_groups:
            if isinstance(group['params'], Variable):
                group['params'] = [group['params']]
            else:
                group['params'] = list(group['params'])
            group_set = set(group['params'])
            if not param_set.isdisjoint(group_set):
                raise ValueError("some parameters appear in more than one "
                                 "parameter group")
            param_set.update(group_set)

        for name, default in defaults.items():
            for i, group in enumerate(self.param_groups):
                if default is required and name not in group:
                    raise ValueError("parameter group " + str(i) + " didn't "
                                                                   "specify a value of required optimization parameter " + name)
                else:
                    group.setdefault(name, default)

        for group in self.param_groups:
            for param in group['params']:
                if not isinstance(param, Variable):
                    raise TypeError("optimizer can only optimize Variables, "
                                    "but one of the params is " + rA9.typename(param))
                if not param.requires_grad:
                    raise ValueError("optimizing a parameter that doesn't "
                                     "require gradients")

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = jnp.full(p.grad.shape,0)

    def step(self, closure):

        raise NotImplementedError
