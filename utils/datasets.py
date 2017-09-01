import os
import csv
import tensorflow as tf


class CSVImageLabelDataset:
    def __init__(self, filename):
        with open(filename) as f:
            self.data = list(zip(*csv.reader(f)))
        base_dir = os.path.dirname(filename) + '/'
        self.single = True
        self.reader = (
            lambda x: tf.read_file(tf.string_join([base_dir, x])),
            lambda x: tf.string_to_number(x, out_type=tf.int32)
        )
        self.out_types = (tf.string, tf.int32)
        self.out_shapes = ((), ())


__all__ = [
    'CSVImageLabelDataset'
]
