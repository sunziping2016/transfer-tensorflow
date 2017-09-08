"""Contains the definition of a base class `Dataset` and some derived classes.
Some ideas are borrowed from `tf.contrib.slim`. However, to be compatible with
`tf.contrib.data`, these classes are necessary.

A Dataset is at least a collection of the following things:
(1) `sources`: a list of objects (e.g. files or tensors) that will be passed to
the `loader`.
(2) `loader`: a function for read and decode.
(3) `multiple`: a bool indicates whether loader reads a batch of samples or only
one sample.
(4) `length`: length of the whole dataset.

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

    def __init__(self, sources, loader=None, multiple=False,
                 length=None, **kwargs):
        """Initializes the dataset.

        Args:
            sources (list): A list of objects.
            loader (function): A function for read and decode.
            multiple (bool): Indicate whether.
            length (int): Length of the whole dataset.
            **kwargs: Any remaining dataset-specific fields.
        """
        kwargs['sources'] = sources
        kwargs['loader'] = loader
        kwargs['multiple'] = multiple
        kwargs['length'] = length
        self.__dict__.update(kwargs)


class ImageFormat(object):
    JPEG = 0
    PNG = 1
    BMP = 2
    GIF = 3


class CSVImageLabelDataset(Dataset):
    """Represents a dataset with a CSV file that has only two colums
    representing image pathes (related to the CSV file) and labels.
    """

    def __init__(self, filename, format=ImageFormat.JPEG,
                 channels=None, start=0):
        """Initializes the dataset.

        Args:
            filename (str): Path to the CSV file.
            format (int): Image format. Defaults to `ImageFormat.JPEG`.
            channels (int): Number of channels of image.
            start (int): Skip first few lines of the CSV file. Defaults to 0.
        """
        with open(filename) as f:
            sources = list(zip(*itertools.islice(csv.reader(f), start, None)))
        self._base_dir = os.path.dirname(filename) + os.path.sep
        sources[0] = list(sources[0])
        sources[1] = list(map(int, sources[1]))
        super(CSVImageLabelDataset, self).__init__(
            sources=sources,
            loader=self._loader,
            multiple=False,
            length=len(sources[0]))
        self.channels = channels
        self.classes = len(sources[0])
        self.format = format
        self._decoder = [
            tf.image.decode_jpeg,
            tf.image.decode_png,
            tf.image.decode_bmp,
            tf.image.decode_gif,
        ][format]

    def _loader(self, image, label):
        image = tf.read_file(tf.string_join([self._base_dir, image]))
        image = self._decoder(image, channels=self.channels)
        return image, label


__all__ = [
    'Dataset',
    'CSVImageLabelDataset'
]
