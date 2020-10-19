from rA9.autograd import Variable


class Spike(Variable):
    def __init__(self, data=None):
        return super(Spike, self).__init__(data, requires_grad=True)

    def __repr__(self):
        return 'Spike containing:' + self.data.__repr__()
