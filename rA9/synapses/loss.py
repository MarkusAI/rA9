import jax.numpy as jnp
from .img2col import *

# LOSS
class Spike_LOSS():
    @staticmethod
    def forward(ctx,output,label,timestep):
        out = 1/2*jnp.sum((output-label)**2)
        e_grad= (output-label)/timestep
        return out ,e_grad

    @staticmethod
    def backward(ctx):
         args= ctx.args
         spike_list=args[0][0]
         weight=arg[1]
         fn, c, fh, fw = weight.shape
         gamma = spike_list[0]

         tau = jnp.divide(jnp.subtract(timestep, spike_list[1]), -self.tau)
         prime = jnp.multiply(jnp.exp(tau), (-1 / self.tau))
         aLIFnet = jnp.multiply(1 / self.Vth, (1 + jnp.multiply(jnp.divide(1, gamma, prime))))
         self.grad = jnp.multiply(weight, aLIFnet)

         self.grad = self.grad.transpose(1, 0).reshape(fn, c, fh, fw)

         d_col = jnp.multiply(weight, aLIFnet)
         dx = col2im_indices(d_col, self.con.shape, fh, fw)

         return LIF_backward(self.tau, self.Vth, dx,
                             spike_list=self.spike_list, e_grad=e_grad,
                             time=timestep), self.weight

