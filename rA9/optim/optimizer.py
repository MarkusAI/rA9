from collections import defaultdict

import rA9
from rA9.autograd import Variable

required = object()


class Optimizer(object):
    """Base class for all optimizers.
    Arguments:
        params (iterable): an iterable of :class:`Variable` s or
            :class:`dict` s. Specifies what Variables should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        if isinstance(params, Variable):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Variables or dicts, but got " +
                            rA9.typename(params))

        self.state = defaultdict(dict)
        self.param_groups = list(params)
        if len(self.param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
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
                                                                   "specify a value of required optimization parameter " +
                                     name)
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
                # if not param.is_leaf:
                #     raise ValueError("can't optimize a non-leaf Variable")

    def __getstate__(self):
        return {
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # p.grad.fill(0)
                    p.grad_fill_zero()

    def step(self, closure):
        """Performs a single optimization step (parameter update).
        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError
