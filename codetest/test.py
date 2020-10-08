import rA9.networks as nn
import rA9.synapses as synapse
import rA9.neurons as neuron

class SCNN(nn.module):

    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1= synapse.Conv2d(1,10,kernel_size=5)
        self.conv2 = synapse.Conv2d(10,20,kernel_size=5)
        self.SNN_IN = neuron.Input(duration=50,dt=1)
        self.LIF = neuron.LIF(tau_m=1,Vth=1,dt=1)
        self.SNN_OUT = neuron.Output(tau_m=1,dt=1)
        self.SNN_IN2 = neuron.Input(duration=50,dt=1)
        self.LIF2 = neuron.LIF(tau_m=1,Vth=1,dt=1)
        self.SNN_OUT2 = neuron.Output(tau_m=1,dt=1)

    def forward(self, x):
        x = synapse.Max_pool2d(self.conv1(x),2)
        x = self.SNN_IN(x)
        x = self.LIF(x)
        x = self.SNN_OUT(x)
        x = synapse.Max_pool2d(self.conv2(x), 2)
        x = self.SNN_IN2(x)
        x = self.LIF2(x)
        x = self.SNN_OUT2(x)

        return x

model = SCNN()
