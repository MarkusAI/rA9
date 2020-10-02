import bintorch.nn as nn
import bintorch.nn.functional as F
from bintorch.autograd import Variable
from examples.data_mnist import MnistDataset, collate_fn
from bintorch.utils.data import DataLoader
import bintorch
import autograd.numpy as np

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 30
batch_size = 64
learning_rate = 0.01

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

train_loader = DataLoader(dataset=MnistDataset(training=True),
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  batch_size=batch_size)

test_loader = DataLoader(dataset=MnistDataset(training=False),
                                  collate_fn=collate_fn,
                                  shuffle=False,
                                  batch_size=batch_size)

optimizer = bintorch.optim.Adam(model.parameters(), lr=learning_rate)


def eval_model(epoch):
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
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images, requires_grad=True)
        labels = Variable(labels)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.data), end=' ')
    eval_model(epoch)