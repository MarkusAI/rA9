from .module import Module
from .. import functional as F


class Dropout(Module):

    def __init__(self, p=0.2, training=True):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.training = training
    def forward(self, input):
        return F.dropout(input, self.p, self.training)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) \
            + inplace_str + ')'
    
