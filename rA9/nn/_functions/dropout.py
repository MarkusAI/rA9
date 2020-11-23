from jax import lax
import jax.numpy as jnp
from rA9.autograd import Function
from rA9.autograd import Variable

class Dropout(Function):
    id = "Dropout"
    @staticmethod
    def forward(ctx, input, p=0.5, train=False):
        assert isinstance(input, Variable)

        def np_fn(input_np, noise):
            return input_np * noise

        noise = binomial(1, p, shape=input.data.shape)
        if not train:
            noise.fill(1)
        if p == 1:
            noise.fill(0)
        np_args = (input.data, noise)
        id = "Dropout"
        return np_fn, np_args, np_fn(*np_args),id

    @staticmethod
    def backward(ctx, grad_output):
        return super(Dropout, Dropout).backward(ctx, grad_output)

    
def binomial(key, p, n=1, shape=()):
    p, n = _promote_shapes(p, n)
    shape = shape or lax.broadcast_shapes(np.shape(p), np.shape(n))
    n_max = np.max(n)
    uniforms = random.uniform(key, shape + (n_max,))
    n = np.expand_dims(n, axis=-1)
    p = np.expand_dims(p, axis=-1)
    mask = (np.arange(n_max) > n).astype(uniforms.dtype)
    p, uniforms = promote_shapes(p, uniforms)
    return np.sum(mask * lax.lt(uniforms, p), axis=-1, keepdims=False)
