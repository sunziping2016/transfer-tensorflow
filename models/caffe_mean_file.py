import numpy as np
import os

mean_files = {
    'ilsvrc_2012': 'ilsvrc_2012_mean.npy',
    'resnet': 'resnet_mean.npy'
}


def mean_file_loader(mean_file):
    return np.load(os.path.join(os.path.dirname(__file__), mean_files[mean_file]))

__all__ = [
    'mean_file_loader'
]
