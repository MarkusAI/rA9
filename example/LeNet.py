import rA9.nn as nn
from rA9.optim import SGD
import rA9.nn.functional as F
from rA9.autograd import Variable
from rA9.utils.data import DataLoader
from rA9.datasets.mnist import MnistDataset, collate_fn



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.input = nn.Input()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding=2)
        self.active1 = nn.LIF()
        self.pool1 = nn.Pooling(size=2, channel=20)
        self.active2 = nn.LIF()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, padding=2)
        self.active3 = nn.LIF()
        self.pool2 = nn.Pooling(size=2, channel=50)
        self.active4 = nn.LIF()
        self.fc1 = nn.Linear(out_features=200,in_features=50*7*7)
        self.active5 = nn.LIF()
        self.fc2 = nn.Linear(out_features=10,in_features=200)
        self.output = nn.Output()
    def forward(self, x, time):
        x = self.input(x)
        x = self.conv1(x)
        x = self.active1(x, time)
        x = self.pool1(x)
        x = self.active2(x, time)
        x = self.conv2(x)
        x = self.active3(x, time)
        x = self.pool2(x)
        x = self.active4(x, time)
        x = x.view(x.data.shape[0], -1)
        x = self.fc1(x)
        x = self.active5(x, time)
        x = self.fc2(x)
        return self.output(x, time)


LeNet = LeNet()
LeNet.train()

PeDurx = 35
batch_size = 16


optimizer = SGD(LeNet.parameters(), lr=0.002)
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
        optimizer.zero_grad()
        for timestep in range(1,PeDurx+1):
            output = LeNet(data, timestep)
        loss = F.Spikeloss(output, target, time_step=PeDurx)
        print("Epoch:" + str(epoch) + " Time: " + str(i) + " loss: " + str(loss.data))
        loss.backward()  # calc gradients
        optimizer.step()  # update gradients
