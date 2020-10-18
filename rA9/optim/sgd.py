import jax.numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    """Implements SpikeSGD algorithm from paper enabling spike-based error backpropagatioin for training deep learning architecture"""

    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                spike_list = p.spike_list
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1
                d_val = grad * spike_list
                p.data += - group['lr'] * d_val

        return loss
