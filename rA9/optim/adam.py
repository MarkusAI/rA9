import math
from .optimizer import Optimizer
import jax.numpy as jnp
import jax.lax as jmath

class Adam(Optimizer):

    def __init__(self, params,spikes, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam, self).__init__(params,spikes, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group,sgroup in zip(self.param_groups,self.spike_groups):
            for p,s in zip(group['params'],sgroup['spikes']):
                if p.grad is None:
                    continue
                grad = jnp.matmul(p.grad,s.data)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = jnp.zeros(grad.shape)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = jnp.zeros(grad.shape)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad += group['weight_decay'] * p.data


                # Decay the first and second moment running average coefficient
                exp_avg = exp_avg * beta1 + (1 - beta1) * grad
                exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * grad * grad
                denom = jnp.sqrt(exp_avg_sq) + group['eps']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * jmath.sqrt(bias_correction2) / bias_correction1
                p.data += -step_size * exp_avg / denom

        return loss
