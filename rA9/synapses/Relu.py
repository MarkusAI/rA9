from ..networks.module import Module
from ._functions import *


class Relu(Module):

    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, input):
        return ReLU.apply(input)
