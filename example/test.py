import rA9.nn as nn
from rA9.optim import Adam
from rA9.autograd import Variable
from rA9.nn.modules import Module
import rA9.nn.functional as F
from example.data_mnist import MnistDataset, collate_fn
from rA9.utils.data import DataLoader
from rA9.utils import PoissonEncoder

batch_size = 16


class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.pool1 = nn.Pooling(2)
        self.pool2 = nn.Pooling(2)
        self.fc1 = nn.Linear(out_features=50, in_features=320)
        self.fc2 = nn.Linear(out_features=10, in_features=50)
        self.output = nn.Output(out_features=10)

    def forward(self, x, time=1):
        x, time = self.conv1(x, time)
        x, time = self.pool1(x,time)
        x, time = self.conv2(x, time)
        x,time = self.pool2(x,time)
        x = Variable(x.data.reshape(-1,320))
        x, time = self.fc1(x, time)
        x, time = self.fc2(x, time)
        return self.output(x, time)


model = SNN()

train_loader = DataLoader(dataset=MnistDataset(training=True, flatten=False),
                          collate_fn=collate_fn,
                          shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False, flatten=False),
                         collate_fn=collate_fn,
                         shuffle=False,
                         batch_size=batch_size)

optimizer = Adam(model.parameters())
model.train()
train_loss = []


for epoch in range(15):
    pe = PoissonEncoder(duration=50)
    model.train()
    for i,(data, target) in enumerate(train_loader):
        target = Variable(target)
        for t, q in enumerate(pe.Encoding(data)):
            data = Variable(q)
            optimizer.zero_grad()
            output = model(data)
            loss = F.Spikeloss(output, target, time_step=t + 1)
            loss.backward()  # calc gradients
            train_loss.append(loss.data)
            optimizer.step()  # update gradients
        if i % 1 == 0:
            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss.data))
        i += 1
