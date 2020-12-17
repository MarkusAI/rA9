import seaborn
import random as rd
import rA9.nn as nn
import jax.numpy as jnp
from rA9.optim import SGD
import jax.random as random
import rA9.nn.functional as F
from rA9.autograd import Variable
from rA9.nn.modules import Module
from rA9.utils.data import DataLoader
from rA9.utils.encoding import PoissonEncoder
from rA9.datasets.mnist import MnistDataset, collate_fn
import os

class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.active1 = nn.LIF()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.active2 = nn.LIF(Vth=0.75)
        self.pool1 = nn.Pooling(size=2, channel=10)
        self.active3 = nn.LIF()
        self.pool2 = nn.Pooling(size=2, channel=20)
        self.active4 = nn.LIF(Vth=0.75)
        self.fc1 = nn.Linear(out_features=100, in_features=320)
        self.active5 = nn.LIF()
        self.fc2 = nn.Linear(out_features=50, in_features=100)
        self.active6 = nn.LIF()
        self.fc3 = nn.Linear(out_features=30, in_features=50)
        self.active7 = nn.LIF()
        self.fc4 =nn.Linear(out_features=10, in_features=30)
        self.output = nn.Output(out_features=10)

    def forward(self, x, time):
        x = self.active1(self.conv1(x), time)
        x = self.active2(self.pool1(x), time)
        x = self.active3(self.conv2(x), time)
        x = self.active4(self.pool2(x), time)
        x = x.view(-1, 320)
        x = self.active5(self.fc1(x), time)
        x = self.active6(self.fc2(x), time)
        x = self.active7(self.fc3(x), time)
        return self.output(self.fc4(x), time), [self.active1.v_current.data, self.active2.v_current.data,
                                             self.active3.v_current.data, self.active4.v_current.data,
                                             self.active5.v_current.data, self.active6.v_current.data,
                                             self.active7.v_current.data, self.output.v_current.data]


model = SNN()
model.train()

PeDurx = 30
batch_size = 64
optimizer = SGD(model.parameters(), lr=0.003)
encoder = PoissonEncoder(duration=PeDurx)
train_loader = DataLoader(dataset=MnistDataset(training=True, flatten=False),
                          collate_fn=collate_fn, shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False, flatten=False),
                         collate_fn=collate_fn, shuffle=False,
                         batch_size=batch_size)

for epoch in range(15):
    for i, (data, target) in enumerate(train_loader):
        target = Variable(target)
        optimizer.zero_grad()
        for j, q in enumerate(encoder.Encoding(data)):
            pdata = Variable(q, requires_grad=True)
            output, v_current = model(pdata, j+1)
            for k, v in enumerate(v_current):
                os.makedirs(f"image/{str(k + 1)}", exist_ok=True)
                if k < 4:
                    seaborn.heatmap(v[0][0], cbar=False).figure.savefig("image/" + str(k + 1) + "/" + str(j) + ".png")
                else:
                    seaborn.heatmap(v, cbar=False).figure.savefig("image/" + str(k + 1) + "/" + str(j) + ".png")
        optimizer.zero_grad()
        loss = F.Spikeloss(output, target, time_step=PeDurx)
        loss.backward()  # calc gradients
        optimizer.step()  # update gradients


        print("Epoch:" + str(epoch) + " Time: " + str(i) + " loss: " + str(loss.data))
