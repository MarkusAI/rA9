from ..networks.variable import Variable


class Parameter(Variable):

    def __init__(self, data=None, requires_grad=True):
        return super(Parameter, self).__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
