"""Contains preprocess classes that can easily chained like layers. The
interface is quite the same as that of PyTorch.

Examples:
    transform = Compose([
        Normalize(mean, std),
        Scale(256),
        RandomCrop(224),
        RandomHorizontalFlip()
    ], 'Preprocess')

    transformed_image = transform(image)

Todo:
    * Some other transforms involving padding, HSL etc.
    * Consider generating multiple images from one image?
"""
import tensorflow as tf


class Compose(list):
    """Composes several transforms together."""

    def __init__(self, transforms, name=None):
        """Initializes the transform.

        Args:
            transforms (list): List of transforms to compose.
            name (str): variable scope for these transforms. Defaults to no
                scope
        """
        super(Compose, self).__init__(transforms)
        self.name = name

    def __call__(self, x):
        if self.name is not None:
            with tf.variable_scope(self.name):
                for layer in self:
                    x = layer(x)
        else:
            for layer in self:
                x = layer(x)
        return x


class Scale(object):
    """Rescales the image to the given size."""

    def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
        """Initializes the transform.

        Args:
            size (int or list): New size of the image.
            method: see `tf.image.resize_images`.
            align_corners:  see `tf.image.resize_images`.
        """
        self.size = size if hasattr(size, '__iter__') else (size,) * 2
        self.method = method
        self.align_corners = align_corners

    def __call__(self, x):
        return tf.image.resize_images(x, self.size, self.method,
                                      self.align_corners)


class CenterCrop(object):
    """Crops the image at the center."""

    def __init__(self, size):
        """Initializes the transform.

        Args:
            size (int or list): New size of the image.
        """
        self.size = size if hasattr(size, '__iter__') else (size,) * 2

    def __call__(self, x):
        # noinspection PyArgumentList
        return tf.image.resize_image_with_crop_or_pad(x, *self.size)


class RandomCrop(object):
    """Crops the image at the Random location."""

    def __init__(self, size):
        """Initializes the transform.

        Args:
            size (int or list): New size of the image.
        """
        self.size = size if hasattr(size, '__iter__') else (size,) * 2

    def __call__(self, x):
        return tf.random_crop(x, self.size + (tf.shape(x)[-1],))


class RandomHorizontalFlip(object):
    """Horizontally flips the given image randomly with a probability of 0.5."""

    def __call__(self, x):
        return tf.image.random_flip_left_right(x)


class Normalize(object):
    """Normalizes an image with mean and standard deviation"""

    def __init__(self, mean, std=None):
        """Initializes the transform.

        Args:
            mean: mean that will be subtracted from image.
            std: std that the image will be divided by. Defaults to no division.
        """
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = tf.subtract(x, self.mean, name='mean')
        if self.std is not None:
            tf.div(x, self.std, name='std')
        return x


__all__ = [
    'Compose',
    'Scale',
    'CenterCrop',
    'RandomCrop',
    'RandomHorizontalFlip',
    'Normalize'
]
