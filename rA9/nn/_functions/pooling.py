from jax import jit
from .img2col import *
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable


# if we use this modules, we need to have a weights, so I had multipled with weights and those ones
class Pooling(Function):
    id = "Pooling"

    @staticmethod
    def forward(ctx, input, weight,v_current,time_step,tau_m,Vth,dt ,size, stride=1):
        assert isinstance(input, Variable)
        assert isinstance(weight, Variable)

        def np_fn(input_np, weight, size):
            return pool_forward(input_np, weight, size=size, stride=stride)

        np_args = (input.data, weight.data, size)
        inv_current = np_fn(*np_args)
        spike_list, v_current = jit(jnp_fn)(x=inv_current,
                                            v_current=v_current.data,
                                            tau_m=tau_m.data,
                                            Vth=Vth.data,
                                            dt=dt.data)

        np_args = (input.data, weight.data, v_current.data, gamma.data, tau_m, Vth, dt)
        grad_np_args = (weights.data, time_step, spike_list, Vth, gamma.data, tau_m)
        return np_fn, np_args, np_fn(*np_args), grad_np_args

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Pooling, Pooling).backward(ctx, grad_outputs)


def pool_forward(X, W, size=2, stride=2):
    n, d, h, w = X.shape
    X = jnp.multiply(X, W)
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
