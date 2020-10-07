from rA9.neurons.LIF import LIF


def LIF_recall(tau,Vth,dt,x,v_current):
    model= LIF(tau_m=tau,Vth=Vth,dt=dt)
    spike_list,v_current=model.forward(x,v_current=v_current)
    return spike_list,v_current



def LIF_backward(tau,Vth,x,spike_list):
    model = LIF(tau_m=tau,Vth=Vth)

    return model.backward(time,spike_list=spike_list,weights=x,e_gradient=)

