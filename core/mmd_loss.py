import tensorflow as tf


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, sigma=None):
    n = tf.shape(source)[0] + tf.shape(target)[0]
    total = tf.concat([source, target], axis=0)
    square = tf.reshape(tf.reduce_sum(tf.square(total), axis=-1), [-1, 1])
    distance = square - 2 * tf.matmul(total, tf.transpose(total)) + tf.transpose(square)
    bandwidth = tf.reduce_sum(distance) / (n * (n - 1)) if sigma is None else tf.constant(sigma, dtype=tf.float32)
    bandwidth_list = [bandwidth * (kernel_mul ** (i - kernel_num // 2)) for i in range(kernel_num)]
    kernel_val = [tf.exp(-distance / i) for i in bandwidth_list]
    return sum(kernel_val)


def mmd_loss(source, target, sampler=None, kernel_mul=2.0, kernel_num=5, sigma=None):
    source_num, target_num = tf.shape(source)[0], tf.shape(target)[0]
    sample_num = tf.maximum(source_num, target_num)
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, sigma=sigma)
    if sampler:
        s1, s2 = sampler(sample_num, source_num)
        t1, t2 = [i + source_num for i in sampler(sample_num, target_num)]
        return tf.reduce_sum(tf.gather_nd(kernels, tf.stack(s1, s2))
                             + tf.gather_nd(kernels, tf.stack(t1, t2))
                             - tf.gather_nd(kernels, tf.stack(s1, t2))
                             - tf.gather_nd(kernels, tf.stack(s2, t1))) / sample_num
    else:
        return tf.reduce_sum(kernels[:source_num, :source_num]) / tf.square(source_num) \
               + tf.reduce_sum(kernels[source_num:, source_num:]) / tf.square(target_num) \
               - 2 * tf.reduce_sum(kernels[:source_num, source_num:]) / source_num / target_num

