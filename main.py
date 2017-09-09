import os
import argparse
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
from models import *
from methods import *


def train_preprocess(image, label, mean=None, crop=(227, 227)):
    image = tf.subtract(image, mean)
    image = tf.random_crop(image, [*crop, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


def test_preprocess(image, label, mean=None, crop=(227, 227)):
    image = tf.subtract(image, mean)
    image = tf.image.resize_image_with_crop_or_pad(image, *crop)
    return image, label


def main(args):
    # Log
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
                            transforms=(test_transform,), shuffle=False))

    # Variables
    training = slim.variable('train', initializer=True, trainable=False,
                             collections=[tf.GraphKeys.LOCAL_VARIABLES])

    # Initializer
    init = tf.group(tf.global_variables_initializer(), source_init)
    train_init = tf.group(tf.assign(training, True), target_train_init)
    test_init = tf.group(tf.assign(training, False), target_test_init)

    with tf.Session() as sess:
        sess.run(init)
        print('Training')
        sess.run(train_init)
        for i in range(100):
            source_val, target_val = sess.run([source, target])
            print(target_val[0].shape)
            print(target_val[1])
            if i % 10 == 0:
                print('Testing')
                sess.run(test_init)
                target_val = sess.run(target)
                print(target_val[0].shape)
                print(target_val[1])
                print('Training')
                sess.run(train_init)

    # Borrow loss weight arguments from args
    # loss_weights = [float(i) for i in args.loss_weights.split(',') if i]
    #
    # train = tf.placeholder_with_default(False, [], name='train')
    # transforms = Sequential([
    #     ToTensor(3),
    #     Scale(256),
    #     Normalize(mean_file_loader('ilsvrc_2012')),
    #     Conditional(
    #         train,
    #         Sequential([
    #             RandomCrop(227),
    #             RandomHorizontalFlip()
    #         ]),
    #         CentralCrop(227)
    #     ),
    # ], name='Transforms')
    # source, target = [
    #     load_data(CSVImageLabelDataset(d), batch_size=args.batch_size, transforms=(transforms,),
    #               num_readers=1, num_preprocess_threads=1,
    #               name=n) for d, n in zip((args.source, args.target), ('SourceDataProvider', 'TargetDataProvider'))
    # ]
    # # Construct base model
    # base_model = Alexnet(train, fc=-1, pretrained=True)
    # # Prepare input images
    # method = DeepAdaptationNetwork(base_model, 31)
    # # Losses and accuracy
    # loss, accuracy = method((source[0], target[0]), (source[1], target[1]), loss_weights)
    # # Optimize
    # step = tf.Variable(0, name='global_step', trainable=False)
    # train_ops = [
    #     tf.train.GradientDescentOptimizer(args.learning_rate)
    #         .minimize(loss, global_step=step,
    #                   var_list=list(filter(lambda x: not x.name.startswith('Linear'),
    #                                        tf.global_variables()))),
    #     tf.train.GradientDescentOptimizer(args.learning_rate * 20)
    #         .minimize(loss, global_step=step,
    #                   var_list=list(filter(lambda x: x.name.startswith('Linear'),
    #                                        tf.global_variables())))
    #
    # ]
    # train_op = tf.group(*train_ops)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.queue_runner.start_queue_runners(coord=coord)
    #     for _ in range(args.max_steps):
    #         _, loss_value, accuracy_value, step_value = sess.run([train_op, loss, accuracy, step], feed_dict={train: True})
    #         print('step: %d\tloss: %.3f\taccuracy: %.3f%%' % (step_value, loss_value, accuracy_value / args.batch_size * 100))
    #     coord.request_stop()
    #     coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Initial learning rate.')
    parser.add_argument('--loss_weights', type=str, default='',
                        help='Comma separated list of loss weights.')
    parser.add_argument('--max_steps', type=int, default=2000, help='Number of steps to run trainer.')
    parser.add_argument('--source', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/office/amazon.csv'),
                        help='Source list file of which every lines are space-separated image paths and labels.')
    parser.add_argument('--target', type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data/office/webcam.csv'),
                        help='Target list file with same layout of source list file. '
                             'Labels are only used for evaluation.')
    parser.add_argument('--base_model', type=str, choices=['alexnet'], default='alexnet',
                        help='Basic model to use.')
    parser.add_argument('--method', type=str, choices=['DAN'], default='DAN',
                        help='Algorithm to use.')
    parser.add_argument('--sampler', type=str, choices=['none', 'fix', 'random'], default='random',
                        help='Sampler for MMD and JMMD. (valid only when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--kernel_mul', type=float, default=2.0,
                        help='Kernel multiplier for MMD and JMMD. (valid only when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--kernel_num', type=int, default=5,
                        help='Number of kernel for MMD and JMMD. (valid only when --loss=mmd or --lost=jmmd)')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'transfer-tensorflow/'),
                        help='Directory to put the log data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=lambda _: main(FLAGS), argv=[sys.argv[0]] + unparsed)
