import rA9.nn as nn
from rA9.optim import Adam
from rA9.autograd import Variable
from rA9.nn.modules import Module
import rA9.nn.functional as F

from rA9.utils import PoissonEncoder

batch_size = 1


class SNN(Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(out_features=784, in_features=100)
        self.fc2 = nn.Linear(out_features=100, in_features=10)
        self.output = nn.Output(out_features=10)

    def forward(self, x, time=1):
        x, time = self.fc1(x, time)
        x, time = self.fc2(x, time)
        return self.output(x, time)


model = SNN()

import jax.numpy as np
from torch.utils import data
from torchvision.datasets import MNIST


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=np.float32))


mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

optimizer = Adam(model.parameters())
model.train()
train_loss = []

i = 0
for epoch in range(15):
    pe = PoissonEncoder(duration=50)
    for data, target in training_generator:
        target =Variable(target)
        for t, q in enumerate(pe.Encoding(data)):
            data = Variable(q)
            optimizer.zero_grad()
            output = model(data)
            loss = F.Spikeloss(output, target, time_step=t+1)
            loss.backward()    # calc gradients
            train_loss.append(loss.data)
            optimizer.step()   # update gradients
        if i % 1 == 0:

            print('Train Step: {}\tLoss: {:.3f}'.format(i, loss.data))
        i += 1