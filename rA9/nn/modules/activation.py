from .module import Module
from .. import functional as F
from rA9.nn.parameter import Parameter
from rA9.autograd.variable import Variable
import jax.numpy as jnp


class LIF(Module):
    def __init__(self, key, tau_m=0.1, Vth=1, dt=1):
        super(LIF, self).__init__()

        self.time_step = 1
        self.tau_m = tau_m
        self.Vth = Vth
        self.key = str(key)
        self.dt = dt
        self.gamma = self.including_object(None)
        self.spike_time_list = self.including_object(None)
        self.v_current = self.including_object(None)

    class including_object(object):
        def __init__(self, data):
            self.data = data

        def init_data(self, size):
            self.data = Variable(jnp.zeros(shape=size))
            return self.data

        def save_data(self, new_data):
            self.data = new_data
            return self.data

        def recall_data(self):
            return self.data

    def forward(self, input, time):

        if time == 0:
            v_current = self.v_current.init_data(input.data.shape)
            gamma = self.gamma.init_data(input.data.shape)
            spike_time_list = self.spike_time_list.init_data(input.data.shape)
        else:
            v_current = self.v_current.recall_data()
            gamma = self.gamma.recall_data()
            spike_time_list = self.spike_time_list.recall_data()

        out, v_current_ret, spike_time = F.LIF(input, v_current, self.tau_m, self.Vth, self.dt, spike_time_list, time,
                                               gamma)
        self.update_spike(key=self.key, layer=out)
        self.v_current.save_data(v_current_ret)
        self.spike_time_list.save_data(spike_time)

        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        return F.relu(input)
=======
from jax import jit
import jax.numpy as np
from rA9.autograd import Function
from rA9.autograd import Variable

class LIF(Function):

    @staticmethod
    def forward(ctx, input, v_current, tau_m, Vth, dt, s_time_list, time, gamma):
        assert isinstance(input, Variable)

        def np_fn(input_np, v_current, tau_m, Vth, dt):
            spike = np.greater_equal(
                v_current + np.multiply(
                    np.divide(
                        np.subtract(
                            input_np,
                            v_current
                        ),
                        tau_m
                    ),
                    dt
                ),Vth).astype('int32')
            v_current = np.where( spike == Vth, 0, v_current*np.exp(-1/tau_m))
            return spike, v_current
        def grad_fn(grad_outputs, s_time_list, time, tau_m, gamma, Vth):
            return np.multiply(grad_outputs,
                (1/Vth*(1+np.multiply(1/gamma,np.sum(np.multiply(-1/tau_m, np.exp(time-s_time_list)))))))

        np_args = (input.data, v_current, tau_m, Vth, dt)
        spike, v_current = jit(np_fn)(*np_args); spike_time = spike*time
        s_time_list = np.concatenate((spike_time, s_time_list))
        grad_np_args = (s_time_list, time, tau_m, gamma, Vth)
        return grad_fn, grad_np_args, spike, s_time_list, v_current

    @staticmethod
    def backward(ctx, grad_output):
        return super(LIF, LIF).backward(ctx, grad_output)



