import tensorflow as tf
import pickle
import os


def alexnet(images, train=True, classes=1000, fc=3, pretrained=False):
    """Build the AlexNet model.
    Args:
      images: Images Tensor in NHWC & RGB format (n x 227 x 227 x 3).
      train: True when training. Only affects dropout.
      classes: The number of fc8.
      fc: The number of full connected layer included in this model (-2 ... 3).
      pretrained: If True, returns a pre-trained caffe model.
    Returns:
      output: the last Tensor of AlexNet.
      parameters: a list of Tensors corresponding to the weights and biases of AlexNet.
    """
    if fc < 0:
        fc += 3

    initializer = {
        'conv1/weights': tf.random_normal([11, 11, 3, 96], stddev=1e-2),
        'conv1/biases': tf.zeros([96]),
        'conv2/weights': tf.random_normal([5, 5, 48, 256], stddev=1e-2),
        'conv2/biases': tf.fill([256], 0.1),
        'conv3/weights': tf.random_normal([3, 3, 256, 384], stddev=1e-2),
        'conv3/biases': tf.zeros([384]),
        'conv4/weights': tf.random_normal([3, 3, 192, 384], stddev=1e-2),
        'conv4/biases': tf.fill([384], 0.1),
        'conv5/weights': tf.random_normal([3, 3, 192, 256], stddev=1e-2),
        'conv5/biases': tf.fill([256], 0.1),
        'fc6/weights': tf.random_normal([9216, 4096], stddev=5e-3),
        'fc6/biases': tf.fill([4096], 0.1),
        'fc7/weights': tf.random_normal([4096, 4096], stddev=5e-3),
        'fc7/biases': tf.fill([4096], 0.1),
        'fc8/weights': tf.random_normal([4096, classes], stddev=5e-3),
        'fc8/biases': tf.zeros([classes]),
    }
    if pretrained:
        params = pickle.load(open(os.path.join(os.path.dirname(__file__), 'caffe_alexnet.pkl'), 'rb'))
        for param in params:
            if param in initializer and initializer[param].shape == params[param].shape:
                initializer[param] = params[param]

    parameters = []
    # conv1
    with tf.variable_scope('conv1'):
        kernel = tf.get_variable('weights', initializer=initializer['conv1/weights'])
        biases = tf.get_variable('biases', initializer=initializer['conv1/biases'])
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # norm1
    with tf.variable_scope('norm1'):
        output = tf.nn.local_response_normalization(output, alpha=2e-5, beta=0.75, depth_radius=2)

    # pool1
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2
    with tf.variable_scope('conv2'):
        output = tf.pad(output, [[0, 0], [2, 2], [2, 2], [0, 0]])
        kernel = tf.get_variable('weights', initializer=initializer['conv2/weights'])
        biases = tf.get_variable('biases', initializer=initializer['conv2/biases'])
        conv = tf.concat([tf.nn.conv2d(*group, [1, 1, 1, 1], padding='VALID')
                          for group in zip(tf.split(output, 2, axis=3), tf.split(kernel, 2, axis=3))], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # norm2
    with tf.variable_scope('norm2'):
        output = tf.nn.local_response_normalization(output, alpha=2e-5, beta=0.75, depth_radius=2)

    # pool2
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # conv3
    with tf.variable_scope('conv3'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]])
        kernel = tf.get_variable('weights', initializer=initializer['conv3/weights'])
        biases = tf.get_variable('biases', initializer=initializer['conv3/biases'])
        conv = tf.nn.conv2d(output, kernel, [1, 1, 1, 1], padding='VALID')
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # conv4
    with tf.variable_scope('conv4'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]])
        kernel = tf.get_variable('weights', initializer=initializer['conv4/weights'])
        biases = tf.get_variable('biases', initializer=initializer['conv4/biases'])
        conv = tf.concat([tf.nn.conv2d(*group, [1, 1, 1, 1], padding='VALID')
                          for group in zip(tf.split(output, 2, axis=3), tf.split(kernel, 2, axis=3))], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # conv5
    with tf.variable_scope('conv5'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]])
        kernel = tf.get_variable('weights', initializer=initializer['conv5/weights'])
        biases = tf.get_variable('biases', initializer=initializer['conv5/biases'])
        conv = tf.concat([tf.nn.conv2d(*group, [1, 1, 1, 1], padding='VALID')
                          for group in zip(tf.split(output, 2, axis=3), tf.split(kernel, 2, axis=3))], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # pool5
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    # fc6
    if fc > 0:
        with tf.variable_scope('fc6'):
            output = tf.reshape(output, [-1, 9216])
            weights = tf.get_variable('weights', initializer=initializer['fc6/weights'])
            biases = tf.get_variable('biases', initializer=initializer['fc6/biases'])
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(output, weights), biases))
            if train:
                output = tf.nn.dropout(output, 0.5)

    # fc7
    if fc > 1:
        with tf.variable_scope('fc7'):
            weights = tf.get_variable('weights', initializer=initializer['fc7/weights'])
            biases = tf.get_variable('biases', initializer=initializer['fc7/biases'])
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(output, weights), biases))
            if train:
                output = tf.nn.dropout(output, 0.5)

    # fc8
    if fc > 2:
        with tf.variable_scope('fc8'):
            weights = tf.get_variable('weights', initializer=initializer['fc8/weights'])
            biases = tf.get_variable('biases', initializer=initializer['fc8/biases'])
            output = tf.nn.bias_add(tf.matmul(output, weights), biases)

    return output, parameters
