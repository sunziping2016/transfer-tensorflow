import tensorflow as tf
import pickle
import os
from utils.layers import *


def alexnet(images, train, fc=3, pretrained=False, caffe_initializer=False):
    """Build the AlexNet model.
    Args:
      images: Images Tensor in NHWC & RGB format (n x 227 x 227 x 3).
      train: True when training. Only affects dropout.
      fc: The number of full connected layer included in this model (-2 ... 3).
      pretrained: If True, returns a pre-trained caffe model.
      caffe_initializer: If True, use Caffe's Alexnet initializer, valid only when pretrained is False
    Returns:
      output: the last Tensor of AlexNet.
      parameters: a list of Tensors corresponding to the weight and bias of AlexNet.
    """
    if fc < 0:
        fc += 3

    init = {
        'conv1/weight': tf.random_normal_initializer(stddev=1e-2),
        'conv1/bias': tf.zeros_initializer,
        'conv2/weight': tf.random_normal_initializer(stddev=1e-2),
        'conv2/bias': tf.constant_initializer(0.1),
        'conv3/weight': tf.random_normal_initializer(stddev=1e-2),
        'conv3/bias': tf.zeros_initializer,
        'conv4/weight': tf.random_normal_initializer(stddev=1e-2),
        'conv4/bias': tf.constant_initializer(0.1),
        'conv5/weight': tf.random_normal_initializer(stddev=1e-2),
        'conv5/bias': tf.constant_initializer(0.1),
        'fc6/weight': tf.random_normal_initializer(stddev=5e-3),
        'fc6/bias': tf.constant_initializer(0.1),
        'fc7/weight': tf.random_normal_initializer(stddev=5e-3),
        'fc7/bias': tf.constant_initializer(0.1),
        'fc8/weight': tf.random_normal_initializer(stddev=5e-3),
        'fc8/bias': tf.zeros_initializer,
    }
    if pretrained:
        params = pickle.load(open(os.path.join(os.path.dirname(__file__), 'caffe_alexnet.pkl'), 'rb'))
        for param in params:
            if param in init:
                init[param] = params[param]
    elif caffe_initializer:
        for param in init:
            init[param] = None

    net = Sequential([
        Sequential([
            Conv2D(3, 96, 11, 4,
                   kernel_initializer=init['conv1/weight'],
                   bias_initializer=init['conv1/bias']),
            ReLU(),
        ], name='conv1'),
        LocalResponseNormalization(2, 1, 2e-5, 0.75, name='norm1'),
        MaxPool(3, 2, name='pool1'),
        Sequential([
            Conv2D(96, 256, 5, 1, 2, groups=2,
                   kernel_initializer=init['conv2/weight'],
                   bias_initializer=init['conv2/bias']),
            ReLU(),
        ], name='conv2'),
        LocalResponseNormalization(2, 1, 2e-5, 0.75, name='norm2'),
        MaxPool(3, 2, name='pool2'),
        Sequential([
            Conv2D(256, 384, 3, 1, 1,
                   kernel_initializer=init['conv3/weight'],
                   bias_initializer=init['conv3/bias']),
            ReLU(),
        ], name='conv3'),
        Sequential([
            Conv2D(384, 384, 3, 1, 1, groups=2,
                   kernel_initializer=init['conv4/weight'],
                   bias_initializer=init['conv4/bias']),
            ReLU(),
        ], name='conv4'),
        Sequential([
            Conv2D(384, 256, 3, 1, 1, groups=2,
                   kernel_initializer=init['conv5/weight'],
                   bias_initializer=init['conv5/bias']),
            ReLU(),
        ], name='conv5'),
        MaxPool(3, 2, name='pool5'),
        Reshape([-1, 9216])
    ], name='alexnet')

    if fc > 0:
        net.append(Sequential([
            Linear(9216, 4096,
                   weight_initializer=init['fc6/weight'],
                   bias_initializer=init['fc6/bias']),
            ReLU(),
            Dropout(train, 0.5),
        ], name='fc6'))

    if fc > 1:
        net.append(Sequential([
            Linear(4096, 4096,
                   weight_initializer=init['fc7/weight'],
                   bias_initializer=init['fc7/bias']),
            ReLU(name='relu7'),
            Dropout(train, 0.5),
        ], name='fc7'))

    if fc > 2:
        net.append(Sequential([
            Linear(4096, 1000,
                   weight_initializer=init['fc8/weight'],
                   bias_initializer=init['fc8/bias']),
        ], name='fc8'))

    return net(images)
