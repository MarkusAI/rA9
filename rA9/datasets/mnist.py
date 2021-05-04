from __future__ import absolute_import
from __future__ import print_function

import os
import gzip
import array
import struct
import numpy as np
import jax.numpy as jnp
from urllib.request import urlretrieve
from rA9.utils.encoding import PoissonEncoder
from rA9.utils.data import DataLoader, Dataset
from future.standard_library import install_aliases

install_aliases()


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        print('Downloading ' + filename)
        urlretrieve(url, out_file)


def mnist():
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist(flatten=True):
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0

    if flatten:
        return train_images, train_labels, test_images, test_labels
    else:
        return train_images.reshape((train_images.shape[0], 1, 28, 28)), train_labels, \
                   test_images.reshape((test_images.shape[0], 1, 28, 28)), test_labels


def collate_fn(batch):
    images = jnp.asarray([b[0] for b in batch])
    labels = jnp.asarray([b[1] for b in batch])
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
