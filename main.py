import argparse
import sys
import tensorflow as tf

def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
