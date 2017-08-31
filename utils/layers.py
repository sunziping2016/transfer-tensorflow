import tensorflow as tf
from tensorflow.python.training import moving_averages
import numpy as np


def conv2d(x, name, in_channels, out_channels, kernel_size, 
           padding=0, stride=1, bias=True, groups=1):
    with tf.variable_scope(name):
        x = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        n = kernel_size * kernel_size * out_channels
        kernel = tf.get_variable('weights', [kernel_size, kernel_size, 
                                 in_channels // groups, out_channels],
                                 initializer=tf.random_normal_initializer(
                                    stddev=np.sqrt(2.0/n)))
        x = tf.concat([tf.nn.conv2d(*group, [1, stride, stride, 1], padding='SAME')
                          for group in zip(tf.split(x, groups, axis=3), 
                                           tf.split(kernel, groups, axis=3))], 
                          axis=3)
        if bias:
            biases = tf.get_variable('biases', [out_channels],
                                     initializer=tf.zeros_initializer)
            x = tf.nn.bias_add(x, biases)
        return x


def relu(x, leakiness=0.0):
    if leakiness > 0:
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    else:
        return tf.nn.relu(x, name='relu')

def dropout(x, drop_rate=0.5):
    return tf.nn.dropout(x, drop_rate)


def batch_norm2d(x, name, train, train_ops=None):
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]
        beta = tf.get_variable(
            'beta', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable(
            'gamma', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32))
        if train:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)
        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, 0.001)
        y.set_shape(x.get_shape())
        return y


def fc(x, name, in_channels, out_channels, bias=True):
    x = tf.reshape(x, [x.get_shape()[0], -1])
    w = tf.get_variable(
        'weight', [in_channels, out_channels],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    x = tf.matmul(x, w)
    if bias:
        b = tf.get_variable('bias', [out_channels],
                            initializer=tf.constant_initializer())
        x = tf.nn.bias_add(x, b)
    return x