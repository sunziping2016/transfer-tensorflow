import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import tensorflow as tf
import models
import numpy as np
import caffe
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check extracted TensorFlow model against Caffe model')
    parser.add_argument('--prototxt', type=str, default=os.path.join(os.path.dirname(__file__), '../tools/bvlc_alexnet.prototxt'),
                        help='path to the Caffe prototxt')
    parser.add_argument('--model', type=str, default=os.path.join(os.path.dirname(__file__), '../tools/bvlc_alexnet.caffemodel'),
                        help='path to the Caffe model')
    args = parser.parse_args()
    images = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='images')
    train = tf.placeholder_with_default(True, [], name='train')
    output = models.alexnet(images, train, pretrained=True)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    data = np.random.rand(10, 227, 227, 3) * 4 + 150
    sess.run(init)
    result = sess.run(output, feed_dict={
        images: data,
        train: False
    })
    caffe_data = data.transpose([0, 3, 1, 2])[:, ::-1]
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.blobs['data'].reshape(*caffe_data.shape)
    net.blobs['data'].data[:] = caffe_data
    net.forward()
    diff = np.sum((result - net.blobs['fc8'].data) ** 2)
    if diff < 1e-5:
        print('Test Okay')
    else:
        print('Test failed')
        print('  diff: %f' % diff)
