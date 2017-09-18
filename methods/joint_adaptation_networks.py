import tensorflow as tf
import utils.layers as nn
import core.losses as L
from .base_method import BaseMethod
from six.moves import zip_longest


class JointAdaptationNetwork(BaseMethod):
    def __init__(self, base_model, n_class):
        super(JointAdaptationNetwork, self).__init__()
        self.base_model = base_model
        self.feature_dim = base_model.output_dim
        self.n_class = n_class
        self.fcb = nn.Linear(self.feature_dim, 256)
        self.fc = nn.Linear(256, n_class)

    def __call__(self, inputs, labels, loss_weights):
        # loss weight: [cross entropy, jmmd]
        inputs = tf.concat(inputs, axis=0)
        features = self.base_model(inputs)
        features = self.fcb(features)
        logits = self.fc(features)
        source_feature, target_feature = tf.split(features, 2)
        source_logits, target_logits = tf.split(logits, 2)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[0],
                                                                       logits=source_logits,
                                                                       name='xentropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        jmmd_losses = [
            L.mmd_loss([source_feature, source_logits], 
                       [target_feature, target_logits]),
        ]
        loss = sum([w * l if w is not None else l
                    for w, l in zip_longest(loss_weights,
                                            jmmd_losses)]) + cross_entropy_loss
        correct = tf.nn.in_top_k(target_logits, labels[1], 1)
        accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
        return loss, accuracy

__all__ = [
    'JointAdaptationNetwork'
]
