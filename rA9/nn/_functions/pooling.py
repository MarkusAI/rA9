from jax import jit
from .img2col import *
from .lif import jnp_fn
from rA9.autograd import Function
from rA9.autograd import Variable


# if we use this modules, we need to have a weights, so I had multipled with weights and those ones
class Pooling(Function):
    id = "Pooling"

    @staticmethod
    def forward(ctx, input, size, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt, stride=1):
        assert isinstance(input, Variable)
        assert isinstance(weights, Variable)
        assert isinstance(v_current, Variable)
        assert isinstance(gamma, Variable)

        def np_fn(input_np, size, weights_np, v_current_np, gamma_np, tau_m, Vth, dt, stride=2):
            inv_current = pool_forward(input_np, weights_np, size=size, stride=stride)

            spike_list, v_current_n = jit(jnp_fn)(x=inv_current, v_current=v_current_np,
                                                  tau_m=tau_m, Vth=Vth, dt=dt)
            index_add(gamma_np, index[:], spike_list)

            return spike_list, v_current_n,index_add(gamma_np, index[:], spike_list)

        np_args = (input.data, weights.data, v_current.data, gamma.data, tau_m, Vth, dt)
        spike, v_current_n, gamma_np = np_fn(*np_args)
        gamma.data = gamma_np
        v_current.data = v_current_n
        spike_time = jnp.multiply(spike, dt * time_step)
        spike_time = jnp.concatenate((spike.T, spike_time.T), axis=1)
        np_grad_args = (weights.data, time_step, spike_time, Vth, gamma, tau_m)
        return np_fn, np_grad_args, spiket

    @staticmethod
    def backward(ctx, grad_outputs):
        return super(Pooling, Pooling).backward(ctx, grad_outputs)


def pool_forward(X, W, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    W_reshaped = W.reshape(n * d, 1, h, w)

    X_col = im2col_indices(X_reshaped, h, w, padding=0, stride=stride)
    W_col = im2col_indices(W_reshaped, h, w, padding=0, stride=stride)

    max_idx_X = jnp.mean(jnp.sum(X_col, axis=0))
    max_idx_W = jnp.mean(jnp.sum(W_col, axis=0))

    out_X = jnp.array(X_col[max_idx_X, range(max_idx_X.size)])
    out_X = out_X.reshape(h_out, w_out, n, d)
    out_X = jnp.transpose(out_X, (2, 3, 0, 1))

    out_W = jnp.array(W_col[max_idx_W, range(max_idx_W.size)])
    out_W = out_W.reshape(h_out, w_out, n, d)
    out_W = jnp.transpose(out_W, (2, 3, 0, 1))

    out = jnp.matmul(out_W,out_X)
    return out
