import tensorflow as tf


def random_sampler(num, maxval, name=None):
    with tf.variable_scope(name, default_name='RandomSampler'):
        indices1 = tf.random_uniform([num], maxval=maxval, dtype=tf.int32, name='RandomIndices1')
        indices2 = tf.random_uniform([num], maxval=maxval - 1, dtype=tf.int32, name='RandomIndices2')
        indices2 += tf.cast(tf.greater_equal(indices2, indices1), tf.int32)
        return indices1, indices2


def fix_sampler(num, maxval):
    with tf.control_dependencies([tf.assert_less_equal(num, maxval)]):
        return tf.range(num, dtype=tf.int32), tf.concat([tf.range(num - 1, dtype=tf.int32), tf.constant([0])], axis=0)
