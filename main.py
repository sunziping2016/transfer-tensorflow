import os
import argparse
import sys
import tensorflow as tf
from utils.layers import Sequential, Conditional, Linear
from utils.datasets import *
from utils.loader import *
from utils.transforms import *
from models import *
from core import *


def main(args):
    if tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)

    train = tf.placeholder_with_default(False, [], name='train')
    transforms = Sequential([
        ToTensor(3),
        Scale(256),
        Normalize(mean_file_loader('ilsvrc_2012')),
        Conditional(
            train,
            Sequential([
                RandomCrop(227),
                RandomHorizontalFlip()
            ]),
            CentralCrop(227)
        ),
    ], name='Transforms')
    source, target = [
        load_data(CSVImageLabelDataset(d), batch_size=args.batch_size, transforms=(transforms,),
                  name=n) for d, n in zip((args.source, args.target), ('SourceDataProvider', 'TargetDataProvider'))
    ]
    base_model = Alexnet(train, fc=-1, pretrained=True)
    last_fc_source, last_fc_target = Linear(4096, 31), Linear(4096, 31)
    inputs = tf.concat([source[0], target[0]], axis=0)
    features = base_model(inputs)
    source_feature, target_feature = tf.split(features, 2)
    source_logits, target_logits = last_fc_source(source_feature), last_fc_target(target_feature)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=source[1], logits=source_logits, name='xentropy')
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    mmd_loss = multiple_mmd_loss([source_feature, source_logits], [target_feature, target_logits])
    loss = cross_entropy_loss + mmd_loss
    correct = tf.nn.in_top_k(target_logits, target[1], 1)
    accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
    optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.queue_runner.start_queue_runners(coord=coord)
        for _ in range(args.max_steps):
            _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict={train: True})
            print('loss: %s\taccuracy: %s' % (loss_value, accuracy_value))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--max_steps', type=int, default=2000, help='Number of steps to run trainer.')
    parser.add_argument('--source', type=str, default=os.path.join(os.path.dirname(__file__), 'data/office/amazon.csv'),
                        help='Source list file of which every lines are space-separated image paths and labels.')
    parser.add_argument('--target', type=str, default=os.path.join(os.path.dirname(__file__), 'data/office/amazon.csv'),
                        help='Target list file with same layout of source list file. '
                             'Labels are only used for evaluation.')
    parser.add_argument('--base_model', type=str, choices=['alexnet'], default='alexnet',
                        help='Basic model to use.')
    parser.add_argument('--loss', type=str, choices=['none', 'mmd', 'jmmd'], default='mmd',
                        help='Loss to apply for transfer learning.')
    parser.add_argument('--sampler', type=str, choices=['none', 'fix', 'random'], default='random',
                        help='Sampler for MMD and JMMD. (valid only when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--kernel_mul', type=float, default=2.0,
                        help='Kernel multiplier for MMD and JMMD. (valid only when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--kernel_num', type=int, default=5,
                        help='Number of kernel for MMD and JMMD. (valid only when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--log_dir', type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'transfer-tensorflow/'),
        help='Directory to put the log data.')
    args, unparsed = parser.parse_known_args()
    tf.app.run(main=lambda _: main(args), argv=[sys.argv[0]] + unparsed)
