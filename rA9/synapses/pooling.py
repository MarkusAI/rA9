import jax.numpy as jnp
from rA9.networks.module import Module



class Max_pool2d(Module):
    def __init__(self, input, kernel_size, stride):
        super(Max_pool2d, self).__init__()
        self.input = input
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, kernel_size):
        def jnp_fn(input_jnp, kernel_size):
            return _pool_forward(input_jnp, kernel_size,stride)

        self.jnp_args = (input, kernel_size)
        out = jnp_fn(*self.jnp_args)
        return out

    def backward(self, grad_outputs):
        jnp_fn = self.jnp_fn
        jnp_args = self.jnp_args
        indexes = [index for index, need_grad in enumerate(self.needs_input_grad) if need_grad]

        jnp_grad_fn = elementwise_grad(jnp_fn, indexes, grad_outputs)
        grads = jnp_grad_fn(*jnp_args)
        return grads


def _pool_forward(X, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    max_idx = jnp.argmax(X_col, axis=0)
    out = jnp.array(X_col[max_idx, range(max_idx.size)])

    out = out.reshape(h_out, w_out, n, d)
    out = jnp.transpose(out, (2, 3, 0, 1))

    return out
