import tensorflow as tf


def alexnet(images, classes=1000, has_fc=3, pretrained=False):
    """Build the AlexNet model.
    Args:
      images: Images Tensor in NHWC & RGB format (n x 227 x 227 x 3).
      classes: The number of fc8.
      has_fc: 0-3, the number of full connected layer included in this model.
      pretrained: If True, returns a pre-trained caffe model.
    Returns:
      output: the last Tensor of AlexNet.
      parameters: a list of Tensors corresponding to the weights and biases of AlexNet.
    """
    parameters = []
    # conv1
    with tf.name_scope('conv1'):
        kernel = tf.get_variable('weights', initializer=tf.random_normal([11, 11, 3, 96], stddev=1e-2))
        biases = tf.get_variable('biases', initializer=tf.zeros([96]))
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='VALID')
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # norm1
    with tf.name_scope('norm1'):
        norm1 = tf.nn.local_response_normalization(output, alpha=1e-4, beta=0.75, depth_radius=2)

    # pool1
    output = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

    # conv2
    with tf.name_scope('conv2'):
        output = tf.pad(output, [[0, 0], [2, 2], [2, 2], [0, 0]])
        kernel = tf.get_variable('weights', initializer=tf.random_normal([5, 5, 48, 256], stddev=1e-2))
        biases = tf.get_variable('biases', initializer=0.1 * tf.ones([256]))
        conv = tf.concat([tf.nn.conv2d(*group, [1, 1, 1, 1], padding='VALID')
                          for group in zip(tf.split(output, 2, axis=3), tf.split(kernel, 2, axis=3))], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # norm2
    with tf.name_scope('norm2'):
        output = tf.nn.local_response_normalization(output, alpha=1e-4, beta=0.75, depth_radius=2)

    # pool2
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

    # conv3
    with tf.name_scope('conv3'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]])
        kernel = tf.get_variable('weights', initializer=tf.random_normal([3, 3, 256, 384], stddev=1e-2))
        biases = tf.get_variable('biases', initializer=tf.zeros([384]))
        conv = tf.nn.conv2d(output, kernel, [1, 1, 1, 1], padding='VALID')
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # conv4
    with tf.name_scope('conv4'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]])
        kernel = tf.get_variable('weights', initializer=tf.random_normal([3, 3, 192, 384], stddev=1e-2))
        biases = tf.get_variable('biases', initializer=0.1 * tf.ones([384]))
        conv = tf.concat([tf.nn.conv2d(*group, [1, 1, 1, 1], padding='VALID')
                          for group in zip(tf.split(output, 2, axis=3), tf.split(kernel, 2, axis=3))], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # conv5
    with tf.name_scope('conv5'):
        output = tf.pad(output, [[0, 0], [1, 1], [1, 1], [0, 0]])
        kernel = tf.get_variable('weights', initializer=tf.random_normal([3, 3, 192, 256], stddev=1e-2))
        biases = tf.get_variable('biases', initializer=0.1 * tf.ones([256]))
        conv = tf.concat([tf.nn.conv2d(*group, [1, 1, 1, 1], padding='VALID')
                          for group in zip(tf.split(output, 2, axis=3), tf.split(kernel, 2, axis=3))], axis=3)
        output = tf.nn.relu(tf.nn.bias_add(conv, biases))
        parameters += [kernel, biases]

    # pool5
    output = tf.nn.max_pool(output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    #fc6
    if has_fc > 0:
        with tf.name_scope('fc6'):
            output = tf.reshape(output, [-1, 9216])
            weights = tf.get_variable('weights', initializer=tf.random_normal([9216, 4096], stddev=5e-3))
            biases = tf.get_variable('biases', initializer=0.1 * tf.ones([4096]))
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(output, weights), biases))

    #fc7
    if has_fc > 1:
        with tf.name_scope('fc7'):
            weights = tf.get_variable('weights', initializer=tf.random_normal([4096, 4096], stddev=5e-3))
            biases = tf.get_variable('biases', initializer=0.1 * tf.ones([4096]))
            output = tf.nn.relu(tf.nn.bias_add(tf.matmul(output, weights), biases))

    #fc8
    if has_fc > 2:
        with tf.name_scope('fc7'):
            weights = tf.get_variable('weights', initializer=tf.random_normal([4096, classes], stddev=1e-2))
            biases = tf.get_variable('biases', initializer=tf.zeros([4096]))
            output = tf.nn.bias_add(tf.matmul(output, weights), biases)

    return output, parameters
