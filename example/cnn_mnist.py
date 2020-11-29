import rA9.nn as nn
import jax.numpy as jnp
from rA9.optim import SGD
import jax.random as random
import rA9.nn.functional as F
from rA9.autograd import Variable
from rA9.nn.modules import Module
from rA9.utils.data import DataLoader
from rA9.datasets.mnist import MnistDataset, collate_fn


class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.active1 = nn.LIF()
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.active2 = nn.LIF()
        self.pool1 = nn.Pooling(size=2, channel=10)
        self.active3 = nn.LIF()
        self.pool2 = nn.Pooling(size=2, channel=20)
        self.active4 = nn.LIF()
        self.fc1 = nn.Linear(out_features=100, in_features=320)
        self.active5 = nn.LIF()
        self.fc2 = nn.Linear(out_features=50, in_features=100)
        self.active6 = nn.LIF()
        self.fc3 = nn.Linear(out_features=30, in_features=50)
        self.active7 = nn.LIF()
        self.fc4 =nn.Linear(out_features=10, in_features=30)
        self.output = nn.Output(out_features=10)
        self.dr = nn.Dropout(p=0.25)

    def forward(self, x, time):
      f = x.data
      for i in range(time):
        rnum = random.uniform(key=random.PRNGKey(0), shape=f.shape)
        uin = (jnp.abs(f) / 2 > rnum).astype('float32')
        uin = jnp.multiply(uin, jnp.sign(f))
        x = self.active1(self.conv1(Variable(uin,requires_grad=True)), i); x = self.dr(x)
        x = self.active2(self.pool1(x), i); x = self.dr(x)
        x = self.active3(self.conv2(x), i); x = self.dr(x)
        x = self.active4(self.pool2(x), i); x = self.dr(x)
        x = x.view(x.data.shape[0], -1)
        x = self.active5(self.fc1(x), i); x = self.dr(x)
        x = self.active6(self.fc2(x), i); x = self.dr(x)
        x = self.active7(self.fc3(x), i); x = self.dr(x)
        result = self.output(self.fc4(x), i)
      return result 


model = SNN()
model.train()

PeDurx = 50
batch_size = 50
optimizer = SGD(model.parameters(), lr=0.003)

train_loader = DataLoader(dataset=MnistDataset(training=True, flatten=False),
                          collate_fn=collate_fn, shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False, flatten=False),
                         collate_fn=collate_fn, shuffle=False,
                         batch_size=batch_size)


for epoch in range(15):
    for i, (data, target) in enumerate(train_loader):
        target = Variable(target)
        data = Variable(data)
        output = model(data, PeDurx)
        loss = F.Spikeloss(output, target, time_step= PeDurx)
        loss.backward()  # calc gradients
        optimizer.step()  # update gradients
        print("Epoch:" + str(epoch) + " Time: " + str(i) + " loss: " + str(loss.data))
