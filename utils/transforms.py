import tensorflow as tf
from .layers_utils import *


def to_tensor(x, channels=None):
    return tf.image.decode_jpeg(x, channels)


def scale(x, height, width=None, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    if width is None:
        width = height
    return tf.image.resize_images(x, [height, width], method, align_corners)


def central_crop(x, height, width=None):
    if width is None:
        width = height
    return tf.image.resize_image_with_crop_or_pad(x, height, width)


def random_crop(x, height, width=None):
    if width is None:
        width = height
    return tf.random_crop(x, [height, width, 3])


def normalize(x, mean, std=None):
    x -= tf.convert_to_tensor(mean, name='mean')
    if std is not None:
        x /= tf.convert_to_tensor(std, name='std')
    return x

ToTensor = make_layer(to_tensor)
Scale = make_layer(scale)
CentralCrop = make_layer(central_crop)
RandomCrop = make_layer(random_crop)
Normalize = make_layer(normalize)
RandomHorizontalFlip = make_layer(tf.image.random_flip_left_right)

__all__ = [
    'to_tensor',
    'scale',
    'central_crop',
    'random_crop',
    'normalize',
    'ToTensor',
    'Scale',
    'CentralCrop',
    'RandomCrop',
    'Normalize',
    'RandomHorizontalFlip'
]
