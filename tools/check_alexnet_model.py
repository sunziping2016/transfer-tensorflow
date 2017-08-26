import os
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorflow as tf
import models
import numpy as np
import caffe

if __name__ == '__main__':
    images = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name='images')
    training = tf.placeholder_with_default(1.0, shape=[], name='training')
    output, parameters = models.alexnet(images, training, pretrained=True)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    data = np.random.rand(10, 227, 227, 3) * 4 + 150
    sess.run(init)
    result = sess.run(output, feed_dict={
        images: data,
        training: 0
    })
    caffe_data = data.transpose(0, 3, 1, 2)
    net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', caffe.TEST)
    net.blobs['data'].reshape(*caffe_data.shape)
    net.blobs['data'].data[:] = caffe_data
    net.forward()
    assert np.sum((result - net.blobs['fc8'].data) ** 2) < 1e-5
    print('Test Okay')
