from rA9.autograd import Variable


class Parameter(Variable):

    def __init__(self, data=None):
        return super(Parameter, self).__init__(data, requires_grad=True)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
