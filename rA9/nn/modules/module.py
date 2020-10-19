from collections import OrderedDict

import rA9
from ..parameter import Parameter
from ..spike import Spike


class Module(object):
    def __init__(self):
        self._spikes = OrderedDict()
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def forward(self, *input):

        raise NotImplementedError

    def register_spike(self, name, spike):
        if '_spikes' not in self.__dict__:
            raise AttributeError(
                "cannot assign spike before Module.__init__() call")
        if spike is None:
            self._spikes[name] = None
        elif not isinstance(spike, Spike):
            raise TypeError("cannot assign '{}' object to spike '{}' "
                            "(rA9.nn.spike or None required)"
                            .format(rA9.typename(spike), name))

        elif spike.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Variable to spike '{0}'. Model "
                "spikes must be created explicitly. To express '{0}' "
                "as a function of another variable, compute the value in "
                "the forward() method.".format(name))

        else:
            self._spikes[name] = spike

    def register_parameter(self, name, param):
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(rA9.typename(param), name))

        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Variable to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another variable, compute the value in "
                "the forward() method.".format(name))

        else:
            self._parameters[name] = param

    def add_module(self, name, module):

        if hasattr(self, name):
            raise KeyError("attribute already exists '{}'".format(name))
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                rA9.typename(module)))
        self._modules[name] = module

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for spike in self._spikes.values():
            if spike is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                spike.data = fn(spike.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def __call__(self, *input, **kwargs):

        result = self.forward(*input, **kwargs)

        return result

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_spikes' in self.__dict__:
            _spikes = self.__dict__['_spikes']
            if name in _spikes:
                return _spikes[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        spikes = self.__dict__.get("_spikes")
        if isinstance(value, Spike):
            if spikes is None:
                raise AttributeError(
                    "cannot assign spikes before Module.__init__() call")
            # remove_from(self.__dict__, self._buffers, self._modules)
            self.register_spike(name, value)
        elif spikes is not None and name in spikes:
            if value is not None:
                raise TypeError("cannot assign '{}' as spike '{}' "
                                "(rA9.nn.spike or None expected)"
                                .format(rA9.typename(value), name))
            self.register_spike(name, value)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            # remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(rA9.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                # remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(rA9.typename(value), name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not rA9.is_tensor(value):
                        raise TypeError("cannot assign '{}' as buffer '{}' "
                                        "(torch.Tensor or None expected)"
                                        .format(rA9.typename(value), name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        elif name in self._spikes:
            del self._spikes[name]
        else:
            object.__delattr__(self, name)

    def spikes(self):
        for name, spike in self.named_spikes():
            yield spike

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def named_spikes(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        for name, p in self._spikes.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_spikes(memo, submodule_prefix):
                yield name, p

    def named_parameters(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        for name, p in self._parameters.items():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p
        for mname, module in self.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in module.named_parameters(memo, submodule_prefix):
                yield name, p

    def children(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self.named_children():
            yield module

    def named_children(self):

        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):

        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):

        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):

        return self.train(False)
