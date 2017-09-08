"""Contains the definition of `load_dataset` and `load_data`. `load_dataset` is
actually a helper function to construct `tf.contrib.data.Dataset` object from
`utils.datasets.Dataset` object, while `load_data` makes switching to different
datasets easier. This can help a lot when you want to test the model while
training, and you don't want to rebuild the graph.

Example:
    data, (train_init, test_init) = load_data(
        # Contains data augmentation, for training purpose
        load_dataset(train_dataset, shuffle=True, ...),
        # No data augmentation, for testing purpose
        load_dataset(test_dataset, shuffle=False, ...)
    )

    train_op, loss = ... # Consumes data

    with tf.Session() as sess:
        sess.run(train_init)
        for i in range(max_iter):
            sess.run(train_op)
            if i % test_interval == 0:
                sess.run(test_init)
                print(sess.run(loss))
                sess.run(train_init)
"""

import itertools
import tensorflow as tf
import tensorflow.contrib.data as data


def load_dataset(dataset, batch_size=None, transforms=None,
                 shuffle=True, shuffle_buffer_size=None, epochs=None):
    """Shuffles and loads data from dataset, applys specified transforms and
    joins transformed data to mini batches.

    Args:
        dataset: A `utils.datasets.Dataset` object.
        batch_size (int): Size of the mini batch. Defaults to no batch.
        transforms (tuple): A list of function to apply to the parsed tensors.
            Defauts to no transform.
        shuffle (bool): Whether to shuffle the input datasets. Defaults to True.
        shuffle_buffer_size (int): The buffer size of the loaded data for
            shuffling. Defaults to no shuffling buffer.
        epochs (int): The epochs to run before gives exception. Defaults to
            infinity.

    Returns:
        A `tf.contrib.data.Dataset` object.
    """
    sources = tuple(map(tf.convert_to_tensor, dataset.sources))
    if shuffle:
        indices = tf.range(0, tf.shape(dataset.sources[0])[0])
        indices = tf.random_shuffle(indices)
        sources = tuple(map(lambda x: tf.gather(x, indices), dataset.sources))
    sources = data.Dataset.from_tensor_slices(sources).repeat(epochs)
    if dataset.loader is not None:
        sources = dataset.loader(sources)
    if shuffle_buffer_size is not None:
        sources.shuffle(buffer_size=shuffle_buffer_size)
    if transforms is not None and len(transforms) != 0:
        def _mapper(*args):
            return [x if x is None else f(x) for x, f in
                    itertools.zip_longest(args, transforms)]
        sources = sources.map(_mapper)
    if batch_size is not None:
        sources = sources.batch(batch_size)
    return sources


def load_data(*datasets):
    """Load data tensors from one or more `tf.contrib.data.Dataset` objects.

    Args:
        datasets: `tf.contrib.data.Dataset` objects.

    Returns:
        Tensors representing data and init ops corresponding to the datasets.
    """
    iterator = tf.contrib.data.Iterator.from_structure(
        datasets[0].output_types,
        datasets[0].output_shapes)
    return iterator.get_next(), \
        tuple(map(lambda x: iterator.make_initializer(x), datasets))


__all__ = [
    'load_dataset',
    'load_data'
]
