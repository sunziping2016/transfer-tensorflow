import tensorflow as tf


class BaseMethod(object):
    def __init__(self):
        pass

    def __call__(self, inputs, labels, loss_weights):
        raise NotImplementedError()
