import itertools
import tensorflow as tf
import tensorflow.contrib.data as data


def load_data(dataset, batch_size=None, transforms=None, shuffle=True,
              shuffle_buffer_size=None):
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

    Returns:
        A `tf.contrib.data.Dataset` object.
    """
    sources = tuple(map(tf.convert_to_tensor, dataset.sources))
    if shuffle:
        indices = tf.range(0, tf.shape(dataset.sources[0])[0])
        indices = tf.random_shuffle(indices)
        sources = tuple(map(lambda x: tf.gather(x, indices), dataset.sources))
    sources = data.Dataset.from_tensor_slices(sources).repeat()
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


def fetch_data(dataset):
    iterator = dataset.make_initializable_iterator()
    return iterator.get_next(), iterator.initializer


def fetch_switchable_data(*datasets):
    iterator = tf.contrib.data.Iterator.from_structure(
        datasets[0].output_types,
        datasets[1].output_shapes)
    return iterator.get_next(), \
        tuple(map(lambda x: iterator.make_initializer(x), datasets))


__all__ = [
    'load_data',
    'fetch_data',
    'fetch_switchable_data'
]
