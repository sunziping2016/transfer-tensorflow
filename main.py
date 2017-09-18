import os
import argparse
import sys
import tensorflow as tf
from tensorflow.python.training import training_util
from utils import *
from models import *
from methods import *


def configure_learning_rate(args, global_step):
    if args.lr_policy == 'fixed':
        return tf.constant(args.lr, name='fixed_learning_rate')
    elif args.lr_policy == 'inv':
        with tf.variable_scope("InverseTimeDecay"):
            global_step = tf.cast(global_step, tf.float32)
            denom = tf.add(1.0, tf.multiply(args.lr_gamma, global_step))
            return tf.multiply(args.lr, tf.pow(denom, -args.lr_power))
    else:
        raise ValueError('lr_policy [%s] was not recognized',
                         args.lr_policy)


def main(args):
    # Log
    if args.log_dir:
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)

    # Preprocess
    mean = mean_file_loader('ilsvrc_2012')
    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.Normalize(mean),
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip()
    ], 'TrainPreprocess')
    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.Normalize(mean),
        transforms.CenterCrop(227)
    ], 'TestPreprocess')

    # Datasets
    source_dataset = datasets.CSVImageLabelDataset(args.source)
    target_dataset = datasets.CSVImageLabelDataset(args.target)

    # Loaders
    source, (source_init,) = loader.load_data(
        loader.load_dataset(source_dataset, batch_size=args.batch_size,
                            transforms=(train_transform,)))
    target, (target_train_init, target_test_init) = loader.load_data(
        loader.load_dataset(target_dataset, batch_size=args.batch_size,
                            transforms=(train_transform,)),
        loader.load_dataset(target_dataset, batch_size=args.batch_size,
                            transforms=(test_transform,)))

    # Variables
    training = tf.get_variable('train', initializer=True, trainable=False,
                               collections=[tf.GraphKeys.LOCAL_VARIABLES])

    # Loss weights
    loss_weights = [float(i) for i in args.loss_weights.split(',') if i]

    # Construct base model
    base_model = Alexnet(training, fc=-1, pretrained=True)

    # Prepare input images
    # method = DeepAdaptationNetwork(base_model, 31)
    method = JointAdaptationNetwork(base_model, 31)

    # Losses and accuracy
    loss, accuracy = method((source[0], target[0]),
                            (source[1], target[1]),
                            loss_weights)

    # Optimize
    global_step = training_util.create_global_step()
    var_list1 = list(filter(lambda x: not x.name.startswith('Linear'),
                            tf.global_variables()))
    var_list2 = list(filter(lambda x: x.name.startswith('Linear'),
                            tf.global_variables()))
    grads = tf.gradients(loss, var_list1 + var_list2)
    learning_rate = configure_learning_rate(args, global_step)
    train_op = tf.group(
        tf.train.MomentumOptimizer(learning_rate, args.momentum)
            .apply_gradients(zip(grads[:len(var_list1)], var_list1)),
        tf.train.MomentumOptimizer(learning_rate * 10, args.momentum)
            .apply_gradients(zip(grads[len(var_list1):], var_list2),
                             global_step=global_step))

    # Initializer
    init = tf.group(tf.global_variables_initializer(), source_init)
    train_init = tf.group(tf.assign(training, True), target_train_init)
    test_init = tf.group(tf.assign(training, False), target_test_init)

    # Run Session
    with tf.Session() as sess:
        sess.run(init)
        sess.run(train_init)
        for _ in range(args.max_steps):
            _, lr_val, loss_val, accuracy_val, step_val = \
                sess.run([train_op, learning_rate, loss, accuracy, global_step])
            if step_val % args.print_freq == 0:
                print('  step: %d\tlr: %.8f\tloss: %.3f\taccuracy: %.3f%%' %
                      (step_val, lr_val, loss_val,
                       float(accuracy_val) / args.batch_size * 100))
            if step_val % args.test_freq == 0:
                accuracies = []
                sess.run(test_init)
                for _ in range(20):
                    accuracies.append(sess.run(accuracy))
                print('test accuracy: %.3f' % (float(sum(accuracies)) /
                                               args.batch_size * 100 / 20.0))
                sess.run(train_init)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Initial learning rate.')
    parser.add_argument('--lr-policy', type=str, choices=['fixed', 'inv'],
                        default='inv',
                        help='Learning rate decay policy.')
    parser.add_argument('--lr-gamma', type=float, default=2e-3,
                        help='Learning rate decay parameter.')
    parser.add_argument('--lr-power', type=float, default=0.75,
                        help='Learning rate decay parameter.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Weight momentum for the solver.')
    parser.add_argument('--loss-weights', type=str, default='',
                        help='Comma separated list of loss weights.')
    parser.add_argument('--max-steps', type=int, default=50000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--source', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'data/office/amazon.csv'),
                        help='Source list file of which every lines are '
                             'space-separated image paths and labels.')
    parser.add_argument('--target', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'data/office/webcam.csv'),
                        help='Target list file with same layout of source list '
                             'file. Labels are only used for evaluation.')
    parser.add_argument('--base-model', type=str, choices=['alexnet'],
                        default='alexnet', help='Basic model to use.')
    parser.add_argument('--method', type=str, choices=['DAN'], default='DAN',
                        help='Algorithm to use.')
    parser.add_argument('--sampler', type=str,
                        choices=['none', 'fix', 'random'],
                        default='random',
                        help='Sampler for MMD and JMMD. (valid only when '
                             '--loss=mmd or --lost=jmmd)')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='')
    parser.add_argument('--test-freq', type=int, default=300,
                        help='')
    parser.add_argument('--kernel-mul', type=float, default=2.0,
                        help='Kernel multiplier for MMD and JMMD. (valid only '
                             'when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--kernel-num', type=int, default=5,
                        help='Number of kernel for MMD and JMMD. (valid only '
                             'when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--log-dir', type=str,default='',
                        help='Directory to put the log data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=lambda _: main(FLAGS), argv=[sys.argv[0]] + unparsed)
