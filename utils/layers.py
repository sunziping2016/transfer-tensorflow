import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.contrib.layers import variance_scaling_initializer


def conv2d(x, in_channels, out_channels, kernel_size, stride=1,
           padding=0, groups=1, bias=True, kernel_initializer=None,
           bias_initializer=tf.zeros_initializer, name=None):
    with tf.variable_scope(name, default_name='Conv2d'):
        if not hasattr(kernel_size, '__iter__'):
            kernel_size = (kernel_size, kernel_size)
        if not hasattr(padding, '__iter__'):
            padding = (padding, padding)
        if not hasattr(stride, '__iter__'):
            stride = (stride, stride)
        if padding[0] != 0 or padding[1] != 0:
            x = tf.pad(x, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
        if kernel_initializer is None:
            kernel_initializer = variance_scaling_initializer(factor=float(groups), mode='FAN_AVG')
        n = kernel_size[0] * kernel_size[1] * out_channels
        kernel = tf.get_variable('weight', [*kernel_size,
                                 in_channels // groups, out_channels],
                                 initializer=kernel_initializer)
        x = tf.concat([tf.nn.conv2d(*group, [1, stride, stride, 1], padding='VALID')
                       for group in zip(tf.split(x, groups, axis=3),
                                        tf.split(kernel, groups, axis=3))],
                      axis=3)
        if bias:
            biases = tf.get_variable('bias', [out_channels],
                                     initializer=bias_initializer)
            x = tf.nn.bias_add(x, biases)
        return x


def relu(x, leakiness=0.0):
    if leakiness > 0.0:
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    else:
        return tf.nn.relu(x, name='relu')


def dropout(x, train, drop_rate=0.5):
    return tf.cond(train, lambda: tf.nn.dropout(x, 1 - drop_rate), lambda: x)


def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x


def fc(x, in_channels, out_channels, bias=True, weight_initializer=None,
       bias_initializer=tf.zeros_initializer, name=None):
    with tf.variable_scope(name, default_name='FullConnected'):
        if weight_initializer is None:
            weight_initializer = variance_scaling_initializer(1.0, mode='FAN_AVG')
        w = tf.get_variable('weight', [in_channels, out_channels],
                            initializer=weight_initializer)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable('bias', [out_channels],
                                initializer=bias_initializer)
            x = tf.nn.bias_add(x, b)
        return x
