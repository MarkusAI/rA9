import jax.numpy as jnp
from rA9.synapses.img2col import *
from rA9.networks.module import Module
from jax import vjp
from jax import linear_util as lu
import jax
from functools import wraps
from jax.api import argnums_partial
import math

# 함수 정의
def elementwise_grad(function, x, initial_gradient=None):
    gradient_function = grad(function, initial_gradient, x)
    return gradient_function


def grad(fun, initial_grad=None, argnums=0):
    value_and_grad_f = value_and_grad(fun, initial_grad, argnums)

    docstr = ("Gradient of {fun} with respect to positional argument(s) "
              "{argnums}. Takes the same arguments as {fun} but returns the "
              "gradient, which has the same shape as the arguments at "
              "positions {argnums}.")

    @wraps(fun, docstr=docstr, argnums=argnums)
    def grad_f(*args, **kwargs):
        ans, g = value_and_grad_f(*args, **kwargs)
        return g

    return grad_f


def value_and_grad(fun, initial_grad=None, argnums=0):
    docstr = ("Value and gradient of {fun} with respect to positional "
              "argument(s) {argnums}. Takes the same arguments as {fun} but "
              "returns a two-element tuple where the first element is the value "
              "of {fun} and the second element is the gradient, which has the "
              "same shape as the arguments at positions {argnums}.")

    @wraps(fun, docstr=docstr, argnums=argnums)
    def value_and_grad_f(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args)
        ans, vjp_py = vjp(f_partial, *dyn_args)

        g = vjp_py(jnp.ones((), jnp.result_type(ans)) if initial_grad is None else initial_grad)
        g = g[0] if isinstance(argnums, int) else g
        return (ans, g)

    return value_and_grad_f


# 대가리 깨져도 elementgradient
class Conv2d(Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = jnp.zeros((self.out_channels, input_channels) + self.kernel_size)

        self.bias = jnp.zeros((output_channels, 1))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.input_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

        keyW = jax.random.PRNGKey(0)
        keyB = jax.random.PRNGKey(0)
        self.weight = jax.random.uniform(minval=-stdv, maxval=stdv, shape=self.weight, key=keyW)
        if self.bias is not None:
            self.bias = jax.random.uniform(minval=-stdv, maxval=stdv, shape=self.bias, key=keyB)

    def forward(self, input):

        def jnp_fn(input_jnp, weights_jnp, bias=None, stride=1, padding=0):

            n_filters, d_filter, h_filter, w_filter = weights_jnp.shape
            n_x, d_x, h_x, w_x = input_jnp.shape
            h_out = (h_x - h_filter + 2 * padding) / stride + 1
            w_out = (w_x - w_filter + 2 * padding) / stride + 1

            if not h_out.is_integer() or not w_out.is_integer():
                raise Exception('Invalid output dimension!')  # 오류 체크
            h_out, w_out = int(h_out), int(w_out)
            X_col = im2col_indices(input_jnp, h_filter, w_filter, padding=padding, stride=stride)
            W_col = weights_jnp.reshape(n_filters, -1)

            out = jnp.matmul(W_col, X_col)

            if b is not None:
                out += b
            out = out.reshape(n_filters, h_out, w_out, n_x)
            out = jnp.transpose(out, (3, 0, 1, 2))

            if bias is None:
                return out
            else:
                return out

        self.jnp_args = (input, self.weights, None if self.bias is None else self.bias)

        output=jnp_fn(*self.jnp_args)

        return output


    def backward(self, grad_outputs):

        jnp_fn = jnp_fn
        jnp_args = self.jnp_args
        indexes = [index for index, need_grad in enumerate(self.needs_input_grad) if need_grad]

        jnp_grad_fn = elementwise_grad(jnp_fn, indexes, grad_outputs)
        grads = jnp_grad_fn(*jnp_args)
        return grads
