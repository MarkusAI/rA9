import jax.numpy as jnp

from ..function import Function


class Add(Function):

    @staticmethod
    def forward(ctx, a, b):
        def jnp_fn(a, b):
            return a + b

        jnp_args = (a.data, b.data)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(Add, Add).backward(ctx, grad_output)


def sort_args(a, b):
    return (a, b, True) if isinstance(a, jnp.ndarray) else (b, a, False)


class View(Function):
    @staticmethod
    def forward(ctx, a, sizes):
        def jnp_fn(a, sizes):
            return jnp.reshape(a, sizes)

        jnp_args = (a.data, sizes)
        return jnp_fn, jnp_args, jnp_fn(*jnp_args)

    @staticmethod
    def backward(ctx, grad_output):
        return super(View, View).backward(ctx, grad_output)
