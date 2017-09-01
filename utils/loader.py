import os
import csv
import tensorflow as tf
from future.moves.itertools import zip_longest
from .layers_utils import make_layer


def load_data(dataset, batch_size, transforms=(), shuffle=True,
              num_readers=4, num_preprocess_threads=4, name=None):
    with tf.name_scope(name, default_name='Loader'):
        item = tf.train.slice_input_producer(dataset.data, shuffle=shuffle)
        sample = [i if r is None else r(i) for i, r in zip_longest(item, dataset.reader)]
        if num_readers > 1:
            if shuffle and not dataset.single:
                data_queue = tf.RandomShuffleQueue(dataset.min_random_capacity + 2 * batch_size * num_readers,
                                                   dataset.min_random_capacity, dataset.out_types, dataset.out_shapes)
            else:
                data_queue = tf.FIFOQueue(2 * batch_size * num_readers, dataset.out_types, dataset.out_shapes)
            enqueue_op = (data_queue.enqueue if dataset.single else data_queue.enqueue_many)(sample)
            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(data_queue, [enqueue_op] * num_readers))
            sample = data_queue.dequeue()
            return tf.train.batch_join(
                [[i if t is None else t(i) for i, t in zip_longest(sample, transforms)]] * num_preprocess_threads,
                batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)
        return sample

DataLoader = make_layer(load_data)

__all__ = [
    'load_data',
    'DataLoader'
]
