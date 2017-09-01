import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.contrib.layers import variance_scaling_initializer
from .layers_utils import *


def conv_nd(x, n, in_channels, out_channels, kernel_size, stride=1, padding=0,
            dilation=1, groups=1, bias=True, kernel_initializer=None,
            bias_initializer=None, name=None):
    with tf.variable_scope(name, default_name='Conv%dd' % n):
        if not hasattr(kernel_size, '__iter__'):
            kernel_size = (kernel_size,) * n
        if not hasattr(padding, '__iter__'):
            padding = (padding,) * n
        if not hasattr(stride, '__iter__'):
            stride = (stride,) * n
        if not hasattr(dilation, '__iter__'):
            dilation = (dilation,) * n
        if padding[0] != 0 or padding[1] != 0:
            x = tf.pad(x, [[0, 0], *zip(padding, padding), [0, 0]])
        if kernel_initializer is None:
            kernel_initializer = variance_scaling_initializer(factor=float(groups), mode='FAN_AVG')
        if bias_initializer is None:
            bias_initializer = tf.zeros_initializer
        kernel = tf.get_variable('weight', [*kernel_size, in_channels // groups,
                                            out_channels] if callable(kernel_initializer) else None,
                                 initializer=kernel_initializer)
        # For better graph layout
        if groups != 1:
            x = tf.concat([
                tf.nn.convolution(*p, 'VALID', stride, dilation, name='conv%dd_%d' % (n, i + 1))
                for i, p in enumerate(zip(tf.split(x, groups, axis=-1, name='split_input'),
                                          tf.split(kernel, groups, axis=-1, name='split_kernel')))
            ], axis=-1)
        else:
            x = tf.nn.convolution(x, kernel, 'VALID', stride, dilation, name='conv%dd' % n)
        if bias:
            biases = tf.get_variable('bias', [out_channels] if callable(kernel_initializer) else None,
                                     initializer=bias_initializer)
            x = tf.nn.bias_add(x, biases)
        return x


def conv1d(x, *args, **kwargs):
    return conv_nd(x, 1, *args, **kwargs)


def conv2d(x, *args, **kwargs):
    return conv_nd(x, 2, *args, **kwargs)


def conv3d(x, *args, **kwargs):
    return conv_nd(x, 3, *args, **kwargs)


def relu(x, leakiness=0.0, name=None):
    if leakiness > 0.0:
        with tf.variable_scope(name, default_name='LeakyRelu'):
            return tf.where(tf.less(x, 0.0), leakiness * x, x)
    else:
        return tf.nn.relu(x, name)


def dropout(x, train, drop_rate=0.5, name=None):
    with tf.variable_scope(name, default_name='Dropout'):
        return tf.nn.dropout(x, 1 - tf.cast(train, tf.float32) * drop_rate)


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
            batch_mean, batch_variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, decay),
                                          assign_moving_average(moving_variance, batch_variance, decay)]):
                return tf.identity(batch_mean), tf.identity(batch_variance)
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


def max_pool(x, kernel_size, strides, name=None):
    if not hasattr(kernel_size, '__iter__'):
        kernel_size = (kernel_size,) * 2
    if not hasattr(strides, '__iter__'):
        strides = (strides,) * 2
    return tf.nn.max_pool(x, [1, *kernel_size, 1], [1, *strides, 1], 'VALID', name=name)


def linear(x, in_channels, out_channels, bias=True, weight_initializer=None,
           bias_initializer=None, name=None):
    with tf.variable_scope(name, default_name='Linear'):
        if weight_initializer is None:
            weight_initializer = variance_scaling_initializer(1.0, mode='FAN_AVG')
        if bias_initializer is None:
            bias_initializer = tf.zeros_initializer
        w = tf.get_variable('weight', [in_channels, out_channels] if callable(weight_initializer) else None,
                            initializer=weight_initializer)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable('bias', [out_channels] if callable(bias_initializer) else None,
                                initializer=bias_initializer)
            x = tf.nn.bias_add(x, b)
        return x


def sequential(x, layers=(), name=None):
    with tf.variable_scope(name) if name is not None else DummyContextMgr():
        for layer in layers:
            x = layer(x)
        return x


class Sequential(list):
    def __init__(self, layers=(), name=None):
        super().__init__(layers)
        self.name = name

    def __call__(self, *args):
        return sequential(*args, layers=self, name=self.name)


def conditional(x, cond, branch1, branch2, name=None):
    return tf.cond(cond, lambda: branch1(x), lambda: branch2(x), name=name)

ConvND = make_layer(conv_nd)
Conv1D = make_layer(conv1d)
Conv2D = make_layer(conv2d)
Conv3D = make_layer(conv3d)
ReLU = make_layer(relu)
Dropout = make_layer(dropout)
BatchNorm = make_layer(batch_norm)
LocalResponseNormalization = make_layer(tf.nn.local_response_normalization)
MaxPool = make_layer(max_pool)
Linear = make_layer(linear)
Reshape = make_layer(tf.reshape)
Conditional = make_layer(conditional)
DoNothing = make_layer(tf.identity)

__all__ = [
    'conv_nd',
    'conv1d',
    'conv2d',
    'conv3d',
    'relu',
    'dropout',
    'batch_norm',
    'sequential',
    'max_pool',
    'linear',
    'conditional',
    'ConvND',
    'Conv1D',
    'Conv2D',
    'Conv3D',
    'ReLU',
    'Dropout',
    'BatchNorm',
    'Linear',
    'Sequential',
    'LocalResponseNormalization',
    'MaxPool',
    'Reshape',
    'Conditional'
]
