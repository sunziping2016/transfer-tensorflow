import os
import argparse
import sys
import tensorflow as tf


def main(args):
    if tf.gfile.Exists(args.log_dir):
        tf.gfile.DeleteRecursively(args.log_dir)
    tf.gfile.MakeDirs(args.log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--source', type=str, default=os.path.join(os.path.dirname(__file__), 'data/office/amazon.txt'),
                        help='Source list file of which every lines are space-separated image paths and labels.')
    parser.add_argument('--target', type=str, default=os.path.join(os.path.dirname(__file__), 'data/office/amazon.txt'),
                        help='Target list file with same layout of source list file. '
                             'Labels are only used for evaluation.')
    parser.add_argument('--base_model', type=str, choices=['alexnet'], default='alexnet',
                        help='Basic model to use.')
    parser.add_argument('--loss', choices=['none', 'mmd'], default='mmd',
                        help='Loss to apply for transfer learning.')
    parser.add_argument('--log_dir', type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'transfer-tensorflow/'),
        help='Directory to put the log data.')
    args, unparsed = parser.parse_known_args()
    tf.app.run(main=lambda: main(args), argv=[sys.argv[0]] + unparsed)
