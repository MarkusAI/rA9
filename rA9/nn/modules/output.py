import jax.numpy as jnp
from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter


class Output(Module):
    def __init__(self):
        pass
    def forward(self, input):
        F.