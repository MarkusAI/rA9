import bintorch.nn as nn
import bintorch.nn.functional as F
from bintorch.autograd import Variable
from examples.data_mnist import MnistDataset, collate_fn
from bintorch.utils.data import DataLoader
import bintorch
import autograd.numpy as np

num_epochs = 30
batch_size = 64
learning_rate = 0.001

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = ConvNet()

train_loader = DataLoader(dataset=MnistDataset(training=True, flatten=False),
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False, flatten=False),
                                  collate_fn=collate_fn,
                                  shuffle=False,
                                  batch_size=batch_size)

optimizer = bintorch.optim.Adam(model.parameters(), lr=learning_rate)


def eval_model(epoch):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for images, labels in test_loader:
        images = Variable(images)
        labels_var = Variable(labels)
        outputs = model(images)
        predicted = np.argmax(outputs.data, 1)
        loss += F.cross_entropy(outputs, labels_var, size_average=False).data
        total += labels.shape[0]
        correct += (predicted == labels).sum()

    print('\rEpoch [{}/{}], Test Accuracy: {}%  Loss: {:.4f}'.format(epoch + 1, num_epochs, 100 * correct / total, loss/total))

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.data), end=' ')
    eval_model(epoch)