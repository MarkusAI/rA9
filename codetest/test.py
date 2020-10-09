import rA9.networks as nn
import rA9.synapses as synapse
import rA9.neurons as neuron


class SCNN(nn.module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.SNN_IN = neuron.Input(duration=50,dt=1)
        self.linear = synapse.Linear(in_features=784,out_features=10)
        self.SNN_OUT = neuron.Output(tau_m=0.1, dt=1)
    def forward(self, x):
        x = self.SNN_IN(x)
        x= self.linear(x)
        x=self.SNN_OUT(x)
        return x


model = SCNN()
