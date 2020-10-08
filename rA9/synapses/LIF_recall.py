from rA9.neurons.LIF import LIF


def LIF_recall(tau,Vth,dt,x,v_current):
    model= LIF(tau_m=tau,Vth=Vth,dt=dt)
    spike_list,v_current=model.forward(x,v_current=v_current)
    return spike_list,v_current


# TIME, EGRAD는 반드시 물어보기
def LIF_backward(tau,Vth,x,spike_list,e_grad,time):
    model = LIF(tau_m=tau,Vth=Vth)

    return model.backward(time=time,spike_list=spike_list,weights=x,e_gradient=e_grad)

