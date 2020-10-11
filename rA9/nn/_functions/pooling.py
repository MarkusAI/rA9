from jax import jit
from .img2col import *
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable


# if we use this modules, we need to have a weights, so I had multipled with weights and those ones
class Pooling(Function):
    id = "Pooling"

    @staticmethod
    def forward(ctx, input, size, stride=1):
        assert isinstance(input, Variable)

        def np_fn(input_np, size):
            return pool_forward(input_np, size=size)

        np_args = (input.data, size)
        return np_fn,np_args,np_fn(*np_args)


    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Pooling, Pooling).backward(ctx, grad_outputs)


def pool_forward(X, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, h, w, padding=0, stride=stride)

    max_idx = jnp.mean(jnp.sum(X_col, axis=0))
    out = jnp.array(X_col[max_idx, range(max_idx.size)])

    out = out.reshape(h_out, w_out, n, d)
    out = jnp.transpose(out, (2, 3, 0, 1))

    return out
