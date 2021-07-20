import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, reskey=2.0):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(jnp.zeros(shape=(out_features, in_features)))
        self.reskey = reskey
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.data.shape
        stdv = jnp.sqrt(self.reskey/size[1])
        self.weight.normal(mean=10, stdv=stdv)

    def forward(self, input):
        out = F.linear(input=input, weights=self.weight)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

