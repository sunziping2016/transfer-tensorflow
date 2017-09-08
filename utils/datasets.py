"""Contains the definition of a base class `Dataset` and some derived classes.
Some ideas are borrowed from `tf.contrib.slim`. However, to be compatible with
`tf.contrib.data`, these classes are necessary.

A Dataset is at least a collection of the following things:
(1) `sources`: a list of objects (e.g. files or tensors) that will be passed to
the `loader`.
(2) `loader` (optional): a function for read and decode that accepts a
`tf.contrib.data.Dataset` and returns a new `tf.contrib.data.Dataset` object.
(3) `length` (optional): length of the whole dataset.

Todo:
    * A image folder dataset
    * A TFRecord dataset
"""

import os
import csv
import itertools
import tensorflow as tf


class Dataset(object):
    """Represents a Dataset specification."""

    def __init__(self, sources, loader=None, length=None, **kwargs):
        """Initializes the dataset.

        Args:
            sources (list): A list of objects.
            loader (function, optional): A function for read and decode.
            length (int, optional): Length of the whole dataset.
            **kwargs: Any remaining dataset-specific fields.
        """
        kwargs['sources'] = sources
        if loader is not None:
            kwargs['loader'] = loader
        if length is not None:
            kwargs['length'] = length
        self.__dict__.update(kwargs)


class CSVImageLabelDataset(Dataset):
    """Represents a dataset with a CSV file that has only two colums
    representing image pathes (related to the CSV file) and labels.
    """

    def __init__(self, filename, format=None, shape=None,
                 channels=None, start=0):
        with open(filename) as f:
            items = itertools.islice(csv.reader(f), start, None)
        self._base_dir = os.path.dirname(filename) + os.path.sep
        sources = list(zip(*items))
        sources[0] = list(sources[0])
        sources[1] = list(map(int, sources[1]))
        super(CSVImageLabelDataset, self).__init__(
            sources=sources,
            loader=self._loader,
            length=len(sources[0]))
        self.channels = channels
        self.shape = shape
        self.classes = len(sources[0])
        self._decoder = tf.image.decode_image if format is None else \
            getattr(tf.image, 'decode_' + format)

    def _loader(self, dataset):
        def _parser(image, label):
            image = tf.read_file(tf.string_join([self._base_dir, image]))
            image = self._decoder(image, channels=self.channels)
            image = tf.image.resize_images(image, self.shape)
            return image, label
        return dataset.flat_map(_parser)


__all__ = [
    'Dataset',
    'CSVImageLabelDataset'
]
