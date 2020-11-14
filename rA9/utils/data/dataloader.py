import rA9
import sys
import traceback
import threading
import collections
import jax.numpy as jnp
import jax.random as random
from jax.ops import index, index_update, index_add
from .sampler import SequentialSampler, RandomSampler, BatchSampler

    
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)


_use_shared_memory = False
"""Whether to use shared memory in default_collate"""


class ExceptionWrapper(object):
    "Wraps an exception plus traceback to communicate across threads"

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))


def _worker_loop(dataset, index_queue, data_queue, collate_fn):
    global _use_shared_memory
    _use_shared_memory = True

    while True:
        r = index_queue.get()
        if r is None:
            data_queue.put(None)
            break
        idx, batch_indices = r
        try:
            samples = collate_fn([dataset[i] for i in batch_indices])
        except Exception:
            data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            data_queue.put((idx, samples))


def _pin_memory_loop(in_queue, out_queue, done_event):
    while True:
        try:
            r = in_queue.get()
        except:
            if done_event.is_set():
                return
            raise
        if r is None:
            break
        if isinstance(r[1], ExceptionWrapper):
            out_queue.put(r)
            continue
        idx, batch = r
        try:
            batch = pin_memory_batch(batch)
        except Exception:
            out_queue.put((idx, ExceptionWrapper(sys.exc_info())))
        else:
            out_queue.put((idx, batch))


def default_collate(batch):


    return batch


class PoissonEncoder(object):
    def __init__(self, duration, dt=1, key=0):
        super().__init__()
        self.dt = dt
        self.duration = duration
        self.key_x = random.PRNGKey(key)

    def Encoding(self, intensities):
        assert jnp.all(intensities >= 0), "Inputs must be non-negative"
        assert intensities.dtype == jnp.float32 or intensities.dtype == jnp.float64, "Intensities must be of type Float."

        # Get shape and size of data.
        shape, size = jnp.shape(intensities), jnp.size(intensities)

        intensities = intensities.reshape(-1)

        time = self.duration // self.dt

        # Compute firing rates in seconds as function of data intensity,
        # accounting for simulation time step.
        rate_p = jnp.zeros(size)
        non_zero = intensities != 0

        rate = index_update(rate_p, index[non_zero], 1 / intensities[non_zero] * (1000 / self.dt))
        del rate_p

        # Create Poisson distribution and sample inter-spike intervals
        # (incrementing by 1 to avoid zero intervals).
        intervals_p = random.poisson(key=self.key_x, lam=rate, shape=(time, len(rate))).astype(jnp.float32)

        intervals = index_add(intervals_p, index[:, intensities != 0],
                              (intervals_p[:, intensities != 0] == 0).astype(jnp.float32))

        del intervals_p

        # Calculate spike times by cumulatively summing over time dimension.

        times_p = jnp.cumsum(intervals, dtype='float32', axis=0)

        times = index_update(times_p, times_p <= time, 0).astype(jnp.int32)
        times = index_update(times, times_p >= time, 1).astype(jnp.bool_)

        del times_p

        return times.reshape(*shape, time)
    


class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        self.collate_fn = loader.collate_fn
        self.batch_sampler = loader.batch_sampler
        self.num_workers = loader.num_workers
        self.pin_memory = loader.pin_memory
        self.done_event = threading.Event()

        self.sample_iter = iter(self.batch_sampler)


    def __len__(self):
        return len(self.batch_sampler)

    def __next__(self):
        if self.num_workers == 0:  # same-process loading
            indices = next(self.sample_iter)  # may raise StopIteration
            batch = self.collate_fn([self.dataset[i] for i in indices])

            return batch

        # check if the next sample has already been generated
        if self.rcvd_idx in self.reorder_dict:
            batch = self.reorder_dict.pop(self.rcvd_idx)
            return self._process_next_batch(batch)

        if self.batches_outstanding == 0:
            self._shutdown_workers()
            raise StopIteration

        while True:
            assert (not self.shutdown and self.batches_outstanding > 0)
            idx, batch = self.data_queue.get()
            self.batches_outstanding -= 1
            if idx != self.rcvd_idx:
                # store out-of-order samples
                self.reorder_dict[idx] = batch
                continue
            return self._process_next_batch(batch)

    next = __next__  #

    def __iter__(self):
        return self

    def _put_indices(self):
        assert self.batches_outstanding < 2 * self.num_workers
        indices = next(self.sample_iter, None)
        if indices is None:
            return
        self.index_queue.put((self.send_idx, indices))
        self.batches_outstanding += 1
        self.send_idx += 1

    def _process_next_batch(self, batch):
        self.rcvd_idx += 1
        self._put_indices()
        if isinstance(batch, ExceptionWrapper):
            raise batch.exc_type(batch.exc_msg)
        return batch

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("DataLoaderIterator cannot be pickled")

    def _shutdown_workers(self):
        if not self.shutdown:
            self.shutdown = True
            self.done_event.set()
            for _ in self.workers:
                self.index_queue.put(None)

    def __del__(self):
        if self.num_workers > 0:
            self._shutdown_workers()


class DataLoader(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, poisson_encoding=False,poisson_encoding_time=110):
        if poisson_encoding is True:
            Pencoder = PoissonEncoder(duration=poisson_encoding_time)
            self.dataset = Pencoder.Encoding(dataset)
        else:
            self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)
