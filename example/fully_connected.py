import rA9.nn as nn
from rA9.optim import SGD
import jax.random as random
import rA9.nn.functional as F
from rA9.autograd import Variable
from rA9.nn.modules import Module
from rA9.utils import PoissonEncoder
from rA9.utils.data import DataLoader
from rA9.datasets.mnist import MnistDataset, collate_fn
import jax.numpy as jnp


class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.input = nn.Input()
        self.fc1 = nn.Linear(in_features=784, out_features=500)
        self.active1 = nn.LIF()
        self.fc2 = nn.Linear(in_features=500, out_features=150)
        self.active2 = nn.LIF()
        self.fc3 = nn.Linear(in_features=150, out_features=50)
        self.active3 = nn.LIF()
        self.fc4 = nn.Linear(in_features=50, out_features=10)
        self.output = nn.Output()

    def forward(self, x, time):
        x = self.input(x,time)
        x = self.active1(self.fc1(x), time)
        x = self.active2(self.fc2(x), time)
        x = self.active3(self.fc3(x), time)
        return self.output(self.fc4(x),time)


model = SNN()
model.train()

PeDurx = 50
batch_size = 64
Pencoder = PoissonEncoder(duration=PeDurx)
optimizer = SGD(model.parameters(), lr=0.01)

train_loader = DataLoader(dataset=MnistDataset(training=True, flatten=True),
                          collate_fn=collate_fn, shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False, flatten=True),
                         collate_fn=collate_fn, shuffle=False,
                         batch_size=batch_size)

for i in range(15):
    for i, (image, label) in enumerate(train_loader):
        label = Variable(label)
        for t, j in enumerate(Pencoder.Encoding(image)):
            image = Variable(j,requires_grad=True)
            t = t + 1
            output = model(image, t)
    optimizer.zero_grad()
    loss = F.Spikeloss(output, label, PeDurx)
    print(loss.data)
    loss.backward()
    optimizer.step()
