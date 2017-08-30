import os
import tensorflow as tf


def chained_call(funcs, x):
    for func in funcs:
        x = func(x)
    return x


def batch_input(filenames, labels, batch_size, transform_image=None, transform_label=None,
                shuffle=True, base_dir='.', num_readers=4, num_preprocess_threads=4):
    with tf.name_scope('batch_processing'):
        item = tf.train.slice_input_producer([filenames, labels], shuffle=shuffle)
        if num_readers > 1:
            data_queue = tf.FIFOQueue(4 * batch_size, [tf.string, tf.int32], shapes=[[], []])
            enqueue_op = data_queue.enqueue([tf.WholeFileReader().read(os.path.join(base_dir, item[0]))[1], item[1]])
            tf.train.queue_runner.add_queue_runner(
                tf.train.queue_runner.QueueRunner(data_queue, [enqueue_op] * num_readers))
            sample = data_queue.dequeue()
        else:
            sample = [tf.WholeFileReader().read(item[0])[1], item[1]]
        if transform_image is None:
            transform_image = []
        elif not hasattr(transform_image, '__iter__'):
            transform_image = [transform_image]
        if transform_label is None:
            transform_label = []
        elif not hasattr(transform_label, '__iter__'):
            transform_label = [transform_label]
        return tf.train.batch_join([[chained_call(transform_image, sample[0]),
                                     chained_call(transform_label, sample[1])]] * num_preprocess_threads,
                                   batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)

