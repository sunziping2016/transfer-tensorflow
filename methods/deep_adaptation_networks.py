import tensorflow as tf
import utils.layers as nn
import core.losses as L
from base_method import BaseMethod

class DeepAdaptationNetworks(BaseMethod):
    def __init__(self, base_model, n_class):
        super(DeepAdaptationNetworks, self).__init__()
        self.base_model = base_model
        self.feature_dim = base_model.feature_dim
        self.n_class = n_class
        self.fc_source = nn.Linear(self.feature_dim,
                                        n_class)
        self.fc_target = nn.Linear(self.feature_dim,
                                        n_class)

    def train(self, inputs, label, loss_weights):
        assert hasattr(loss_weights, 'cross_entropy_loss')
        assert hasattr(loss_weights, 'mmd_loss')

        inputs = tf.concat(inputs, axis=0)
        features = self.base_model(inputs)
        source_feature, target_feature = tf.split(features, 2)
        source_logit = self.fc_source(source_feature)
        target_logit = self.fc_target(target_feature)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, 
                                                                       logits=source_logit, 
                                                                       name='xentropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        mmd_loss = L.mmd_loss([source_feature, source_logits],
                              [target_feature, target_logits])
        loss = loss_weights['cross_entropy_loss'] * cross_entropy_loss \
             + loss_weights['mmd_loss'] * mmd_loss
        return loss, target_logit

    def validation(self, input):
        feature = self.base_model(input)
        return self.fc_target(feature)