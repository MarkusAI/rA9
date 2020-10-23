from . import _functions


def Spikeloss(outputs, labels, time_step):
    return _functions.Spikeloss.apply(outputs, labels, time_step)


def linear(input, weights):
    return _functions.Linear.apply(input, weights)


def conv2d(input, weights, stride=1, padding=0):
    return _functions.Conv2d.apply(input, weights, stride, padding)


def Output(input, weights, v_current, tau_m, dt, time_step, Vth, gamma):
    return _functions.Output.apply(input, weights, v_current, tau_m, dt, time_step, Vth, gamma)


def dropout(input, p=0.5, training=False):
    return _functions.Dropout.apply(input, p, training)


def pooling(input, size, weights, stride):
    return _functions.Pooling.apply(input, size, weights, stride)


def LIF(input, v_current, tau_m, Vth, dt, s_time_list, time, gamma):
    return _functions.LIF.apply(input, v_current, tau_m, Vth, dt, s_time_list, time, gamma)
