!pip install rA9



from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
from torch.utils.data import DataLoader, Dataset

install_aliases()

import os
import gzip
import struct
import array
import jax.numpy as np
import numpy as onp

from urllib.request import urlretrieve

def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        print('Downloading ' + filename)
        urlretrieve(url, out_file)

def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return onp.array(array.array("B", fh.read()), dtype=onp.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return onp.array(array.array("B", fh.read()), dtype=onp.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels

def load_mnist(flatten=True):
    partial_flatten = lambda x : onp.reshape(x, (x.shape[0], onp.prod(x.shape[1:])))

    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0

    return train_images if flatten else train_images.reshape((train_images.shape[0], 1, 28, 28)), \
           train_labels, \
           test_images if flatten else test_images.reshape((test_images.shape[0], 1, 28, 28)), \
           test_labels

def collate_fn(batch):
    images = np.asarray([b[0] for b in batch])
    labels = np.asarray([b[1] for b in batch])
    return images, labels

class MnistDataset(Dataset):

    def __init__(self, training=True, flatten=True):
        self.training = training
        self.train_images, self.train_labels, self.test_images, self.test_labels = load_mnist(flatten)

    def __len__(self):
        if self.training:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):

        if self.training:
            return self.train_images[idx], self.train_labels[idx]
        else:
            return self.test_images[idx], self.test_labels[idx]

import rA9.nn as nn
from rA9.utils import tpu
from rA9.optim import Adam
from rA9.autograd import Variable
from rA9.nn.modules import Module
import rA9.nn.functional as F
from rA9.utils.data import DataLoader
from rA9.utils import PoissonEncoder
from jax import jit


batch_size = 64


class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.active1 = nn.LIF(key=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.active2 = nn.LIF(key=2)
        self.pool1 = nn.Pooling(size=2, channel=10)
        self.active3 = nn.LIF(key=3)
        self.pool2 = nn.Pooling(size=2, channel=20)
        self.active4 = nn.LIF(key=4)
        self.fc1 = nn.Linear(out_features=50, in_features=320)
        self.active5 = nn.LIF(key=5)
        self.fc2 = nn.Linear(out_features=10, in_features=50)
        self.active6 = nn.LIF(key=6)
        self.output = nn.Output(out_features=10)

    def forward(self, x, time):
        x = self.active1(self.conv1(x), time)
        x = self.active2(self.pool1(x), time)
        x = self.active3(self.conv2(x), time)
        x = self.active4(self.pool2(x), time)
        x = x.view(-1, 320)
        x = self.active5(self.fc1(x), time)
        x = self.active6(self.fc2(x), time)
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

model.train()
duration = 100
pe = PoissonEncoder(duration=duration)
optimizer = Adam(model.parameters(), lr=0.002)


for epoch in range(15):
  for i, (data, target) in enumerate(train_loader):
    for t, q in enumerate(pe.Encoding(data)):
      F.Spikeloss(model(Variable(q,requires_grad=True), t), Variable(target), time_step=t + 1).backward()
      optimizer.step()
    if i % 1 == 0:
      print('Epoch:{}\tTrain Step: {}\tLoss: {:.3f}'.format(epoch, i, loss.data))
