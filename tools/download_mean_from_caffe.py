import os
import argparse
from tools_utils import download
import pickle


caffe_mean_files = {
    # name: (url, path to file if compression, save path for binaryproto, save path for pkl)
    # From http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    'ilsvrc_2012': (
        'https://file.szp.io/f/1be381bd3c/?dl=1',
        'ilsvrc_2012_mean.binaryproto',
        '../models/ilsvrc_2012_mean.npy'
    ),
    # From https://github.com/KaimingHe/deep-residual-networks
    'resnet': (
        'https://file.szp.io/f/b7eeb02e39/?dl=1',
        'resnet_mean.binaryproto',
        '../models/resnet_mean.npy'
    )
}


def extract_binaryproto(proto, output):
    from caffe.proto.caffe_pb2 import BlobProto
    from caffe.io import blobproto_to_array
    import numpy as np
    blob = BlobProto()
    data = open(proto, 'rb').read()
    blob.ParseFromString(data)
    mean = np.array(blobproto_to_array(blob)).squeeze(0) \
        .transpose([1, 2, 0]).astype(np.float32)
    pickle.dump(mean, open(output, 'wb'), protocol=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Caffe mean file and convert it to NumPy pickle')
    parser.add_argument('--proto', type=str, default='', help='download Caffe binaryproto to or load it from this path')
    parser.add_argument('-o', dest='output', type=str, default='', help='save NumPy pickle to this path')
    parser.add_argument('-m', dest='mean', type=str, choices=list(caffe_mean_files.keys()),
                        default='ilsvrc_2012', help='Mean file to download')
    args = parser.parse_args()
    if len(args.proto) == 0:
        args.proto = os.path.join(os.path.dirname(__file__), caffe_mean_files[args.mean][1])
    if len(args.output) == 0:
        args.output = os.path.join(os.path.dirname(__file__), caffe_mean_files[args.mean][2])
    if not os.path.exists(args.proto):
        print('Downloading binaryproto...')
        download(args.proto, caffe_mean_files[args.mean][0])

    print('Extracting mean file...')
    extract_binaryproto(args.proto, args.output)
