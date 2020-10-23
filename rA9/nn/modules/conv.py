from rA9.autograd import Function
from rA9.autograd import Variable
from jax import numpy as np
from .img2col import *
from jax import grad


class Conv2d(Function):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.weight = Parameter(jnp.zeros((out_channels, in_channels) + self.kernel_size))
        self.stride = stride
        self.padding = padding
        self.reset_parameters()

    def forward(self, input):
        out = F.conv2d(input=input, weights=self.weight, stride=self.stride, padding=self.padding)

        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def conv_forward(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1
    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)
    out = np.matmul(W_col, X_col)
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = np.transpose(out, (3, 0, 1, 2))

    return out


