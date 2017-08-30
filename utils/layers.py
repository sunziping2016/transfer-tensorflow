import tensorflow as tf
import numpy as np


def conv(x, name, in_channels, out_channels, kernel_size, padding='SAME', stride=1, bias=True):
	with tf.variable_scope(name):
		n = kernel_size * kernel_size * out_channels
		kernel = tf.get_variable('weight', 
		                         [kernel_size, kernel_size, in_channels, out_channels],
		                         initializer=tf.random_normal_initializer(
		                            stddev=np.sqrt(2.0/n)))
		x = tf.nn.conv2d(x, kernel, stride, padding=padding)