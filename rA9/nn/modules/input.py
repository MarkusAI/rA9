from .module import Module
from .. import functional as F


class Input(Module):

    def __init__(self):
        super(Input, self).__init__()

    def forward(self, input):
        return F.Input(input)

        

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
