import tensorflow as tf


def to_tensor(channels=None):
    def callback(contents):
        return tf.image.decode_jpeg(contents, channels)
    return callback


def scale(size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    def callback(images):
        return tf.image.resize_images(images, size, method, align_corners)
    return callback


def central_crop(target_height, target_width):
    def callback(image):
        return tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
    return callback


def random_crop(target_height, target_width):
    def callback(image):
        return tf.random_crop(image, [target_height, target_width, 3])
    return callback


def normalize(mean, std=None):
    def callback(image):
        image -= tf.convert_to_tensor(mean, name='mean')
        if std is not None:
            image /= tf.convert_to_tensor(std, name='std')
        return
