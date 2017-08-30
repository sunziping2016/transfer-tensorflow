import tensorflow as tf


def to_tensor(channels=None):
    def hook(contents):
        return tf.image.decode_jpeg(contents, channels)
    return hook


def scale(size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    def hook(images):
        return tf.image.resize_images(images, size, method, align_corners)
    return hook
