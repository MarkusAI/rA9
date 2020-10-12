from . import _functions


def Spikeloss(outputs, labels, time_step):
    return _functions.Spikeloss.apply(outputs, labels, time_step)


def linear(input, weight, v_current, gamma, tau_m, Vth, dt,time_step):
    return _functions.Linear.apply(input, weight, v_current, gamma, tau_m, Vth, dt,time_step)


def conv2d(input, weights, v_current, tau_m, Vth, dt,time_step, stride=1, padding=0):
    return _functions.Conv2d.apply(input, weights, v_current,time_step, tau_m, Vth, dt, stride, padding)


def Output(input, weights, v_current, tau_m, dt, time_step):
    return _functions.Output.apply(input, weights, v_current, tau_m, dt, time_step)


def dropout(input, p=0.5, training=False):
    return _functions.Dropout.apply(input, p, training)


def pooling(input, size,weight,v_current,time_step,tau_m,Vth,dt, stride=1):
    return _functions.Pooling.apply(input,weight, v_current,time_step, tau_m, Vth, dt,size, stride)