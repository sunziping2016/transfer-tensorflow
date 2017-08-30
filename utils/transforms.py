import tensorflow as tf


def to_tensor(channels=None):
    def hook(contents):
        return tf.image.decode_jpeg(contents, channels)
    return hook


def scale(size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    def hook(images):
        return tf.image.resize_images(images, size, method, align_corners)
    return hook


def central_crop(target_height, target_width):
    def hook(image):
        return tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
    return hook


def random_crop(target_height, target_width):
    def hook(image):
        return tf.random_crop([target_height, target_width, 3])
    return hook
