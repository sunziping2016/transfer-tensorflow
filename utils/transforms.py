import tensorflow as tf


def to_tensor(channels=None):
    def callback(contents):
        return tf.image.decode_jpeg(contents, channels)
    return callback


def scale(height, width=None, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    if width is None:
        width = height

    def callback(images):
        return tf.image.resize_images(images, [height, width], method, align_corners)
    return callback


def central_crop(height, width=None):
    if width is None:
        width = height

    def callback(image):
        return tf.image.resize_image_with_crop_or_pad(image, height, width)
    return callback


def random_crop(height, width=None):
    if width is None:
        width = height

    def callback(image):
        return tf.random_crop(image, [height, width, 3])
    return callback


def normalize(mean, std=None):
    def callback(image):
        image -= tf.convert_to_tensor(mean, name='mean')
        if std is not None:
            image /= tf.convert_to_tensor(std, name='std')
        return image
    return callback

__all__ = [
    'to_tensor',
    'scale',
    'central_crop',
    'random_crop',
    'normalize'
]
