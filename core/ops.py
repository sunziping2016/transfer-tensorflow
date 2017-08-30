import tensorflow as tf
from tensorflow.python.framework import ops

def gradient_lr(self, x, low=0.0, high=1.0, max_iter=2000.0, alpha=10.0):
    height = high - low
    progress = tf.minimum(1.0, tf.cast(self.global_step, tf.float32) / max_iter)
    lr_mult = tf.div(2.0*height, (1.0+tf.exp(-alpha*progress))) - height + low
    def gradient_lr_grad(op, grad):
        x_ = op.inputs[0]
        y_ = op.inputs[1]
        return [grad * -y_, tf.zeros([])]
    with ops.name_scope(None, "Gradient", [x, lr_mult]) as name:
        gr_x_y = py_func(lambda x_, y_: x_, 
                         [x, lr_mult], 
                         [tf.float32],
                         name=None,
                         grad=gradient_lr_grad)
        return gr_x_y[0]