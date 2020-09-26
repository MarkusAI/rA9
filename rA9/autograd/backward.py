import jax
import jax.numpy as jnp
from jax import grad
from jax import jit


# Leaky Integrate And Fire Model

    # tau => Time Constant; Capacity * Resistance
    # Vmem => Rest Voltage



class spike_error(object):
    def __init__(self, tau, Vmem, Vth, rate, lr, **kwargs):
        self.spikelayers = kwargs['spikelayers']
        self.tau = tau
        self.Vmem = Vmem
        self.Vth = Vth
        self.rate = rate  # homogeneous post-neuronal firing rate,
        self.learning_rate = lr

    @jit
    def Out_Error(self, pred_y, y):
        return pred_y - y

    @jit
    def expotential_function(self, time, time_keep, keep):
        global ret
        exp = jnp.exp(-1 * (jnp.divide(time - time_keep, self.tau)))
        for i in range(keep):
            ret -= exp
        return ret

    @jit
    def lif_gradient(self, time, time_keep, keep):
        exp = self.expotential_function(time, time_keep, keep)
        grad = jnp.add(jnp.divide(exp, self.rate), 1)
        return jnp.divide(grad, self.Vth)

    @jit
    def backward(self, pred_y, y):
        self.grad = []
        for l in reversed(range(1, len(self.spikelayers))):

            if l == len(self.spikelayers) - 1:
                e = self.Out_Error(pred_y, y)  # final layer (outpur-label)/ tau
                gradient = jnp.divide(e, self.tau)
            else:
                gradient = (weight[l] * grad[-1]) * \
                           self.lif_gradient(inputtime[l], time[l], len(self.spikelayers) - l)
            weight[l] -= self.learning_rate * gradient
            self.grad.append(gradient)
