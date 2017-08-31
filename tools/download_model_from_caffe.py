import os
import sys
import argparse
from utils import download, check

caffemodels = {
    'alexnet': (
        'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt',
        'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
        '9116a64c0fbe4459d18f4bb6b56d647b63920377',
        'bvlc_alexnet.prototxt',
        'bvlc_alexnet.caffemodel',
        '../models/caffe_alexnet.pkl'
    )
}


def extract_model(prototxt, caffemodel, output):
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    import pickle
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    model = {}
    first_conv, first_fc = True, True
    for name in net.params:
        params = net.params[name]
        if name.startswith('fc'):
            if first_fc:
                model[name + '/weight'] = params[0].data \
                    .reshape(4096, 256, 6, 6).transpose(2, 3, 1, 0).reshape(9216, 4096)
                first_fc = False
            else:
                model[name + '/weight'] = params[0].data.transpose(1, 0)
            if len(params) > 1:
                model[name + '/bias'] = params[1].data
        elif name.startswith('conv'):
            if first_conv:
                model[name + '/weight'] = params[0].data[:, ::-1].transpose(2, 3, 1, 0)
                first_conv = False
            else:
                model[name + '/weight'] = params[0].data.transpose(2, 3, 1, 0)
            if len(params) > 1:
                model[name + '/bias'] = params[1].data
        else:
            print('Unknown layer: %s' % name, file=sys.stderr)
    pickle.dump(model, open(output, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download trained Caffe model and convert it to TensorFlow model')
    parser.add_argument('--prototxt', type=str, default='', help='download Caffe prototxt to or load it from this path')
    parser.add_argument('--caffemodel', type=str, default='', help='download Caffe model to or load it from this path')
    parser.add_argument('-o', dest='output', type=str, default='', help='save TensorFlow model to this path')
    parser.add_argument('--model', type=str, choices=list(caffemodels.keys()),
                        default='alexnet', help='Model to download')
    args = parser.parse_args()
    if len(args.prototxt) == 0:
        args.prototxt = os.path.join(os.path.dirname(__file__), caffemodels[args.model][3])
    if len(args.caffemodel) == 0:
        args.caffemodel = os.path.join(os.path.dirname(__file__), caffemodels[args.model][4])
    if len(args.output) == 0:
        args.output = os.path.join(os.path.dirname(__file__), caffemodels[args.model][5])

    if not os.path.exists(args.prototxt):
        print('Downloading prototxt...')
        download(args.prototxt, caffemodels[args.model][0])

    if not os.path.exists(args.caffemodel):
        print('Downloading model...')
        download(args.caffemodel, caffemodels[args.model][1])
        if not check(args.caffemodel, sha1=caffemodels[args.model][2]):
            print('ERROR: model did not download correctly! Run this again.')
            sys.exit(1)

    print('Extracting model...')
    extract_model(args.prototxt, args.caffemodel, args.output)
