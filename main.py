import os
import argparse
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from utils import *
from models import *


def main(args):
    if tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)
    transforms = [
        to_tensor(3),
        scale(256),
        normalize(mean_file_loader('ilsvrc_2012')),
        central_crop(227)
    ]
    batch, classes = batch_input_from_csv(args.source, transform_image=transforms, batch_size=100, shuffle=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.queue_runner.start_queue_runners(coord=coord)
        import numpy as np
        images = sess.run(batch[0]).astype(np.uint8)
        coord.request_stop()
        coord.join(threads)
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(10, 10)
        for i in range(100):
            ax = plt.subplot(gs[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(images[i])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
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
