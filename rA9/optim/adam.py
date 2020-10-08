import math
from .optimizer import Optimizer
import jax.numpy as jnp


class Adam(Optimizer):
    def __init__(self, params,d_weights, lr=0.001, betas=(0.88, 0.999), eps=2e-7, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)
        self.d_weights =d_weights

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                count = 0
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = jnp.zeros(p.shape)
                    state['exp_avg_sq'] = jnp.zeros(p.shape)
                d_w= self.d_weights[count]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg = exp_avg * beta1 + (1 - beta1)*d_w
                exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2)*d_w**2
                denom = jnp.sqrt(exp_avg_sq) + group['eps']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p += -step_size * exp_avg / denom*d_w

        return loss
