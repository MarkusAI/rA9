from . import _functions


def Spikeloss(outputs, labels, time_step):
    return _functions.Spikeloss.apply(outputs, labels, time_step)


def linear(input, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt):
    return _functions.Linear.apply(input, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt)


def conv2d(input, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt, stride=1, padding=0):
    return _functions.Conv2d.apply(input, time_step, weights, spike_list, v_current, gamma, tau_m, Vth, dt, stride, padding)


def Output(input, weights, v_current, tau_m, dt, time_step):
    return _functions.Output.apply(input, weights, v_current, tau_m, dt, time_step)


def dropout(input, p=0.5, training=False):
    return _functions.Dropout.apply(input, p, training)


def pooling(input, size, stride=1):
    return _functions.Pooling.apply(input, size, stride)
