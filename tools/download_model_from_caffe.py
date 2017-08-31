import os
import sys
import argparse
from math import sqrt
from tools_utils import download, check

caffe_models = {
    'alexnet': (
        'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt',
        'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
        '9116a64c0fbe4459d18f4bb6b56d647b63920377',
        'bvlc_alexnet.prototxt',
        'bvlc_alexnet.caffemodel',
        '../models/caffe_alexnet.pkl',
        256 # First fc input channel size
    ),
    'resnet50': (
        'https://file.szp.io/f/d367d253db/?dl=1',
        'https://file.szp.io/f/80094d907e/?dl=1',
        'b7c79ccc21ad0479cddc0dd78b1d20c4d722908d',
        'ResNet-50-deploy.prototxt',
        'ResNet-50-model.caffemodel',
        '../models/caffe_resnet50.pkl',
        2048
    ),
    'resnet101': (
        'https://file.szp.io/f/f6ecb01de7/?dl=1',
        'https://file.szp.io/f/3dda65e477/?dl=1',
        '1dbf5f493926bb9b6b3363b12d5133c0f8b78904',
        'ResNet-101-deploy.prototxt',
        'ResNet-101-model.caffemodel',
        '../models/caffe_resnet101.pkl',
        2048
    ),
    'resnet152': (
        'https://file.szp.io/f/0590506dba/?dl=1',
        'https://file.szp.io/f/97ae78f7ba/?dl=1',
        '251edb93604ac8268c7fd2227a0f15144310e1aa',
        'ResNet-152-deploy.prototxt',
        'ResNet-152-model.caffemodel',
        '../models/caffe_resnet152.pkl',
        2048
    ),
}


def extract_model(prototxt, model, output, first_fc_in):
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    import pickle
    net = caffe.Net(prototxt, model, caffe.TEST)
    model = {}
    first_conv, first_fc = True, True
    for name in net.params:
        params = net.params[name]
        if name.startswith('fc'):
            if first_fc:
                shape = (params[0].data.shape[0], first_fc_in,
                         *(int(sqrt(params[0].data.shape[1] // first_fc_in) + 0.5),) * 2)
                model[name + '/weight'] = params[0].data.reshape(*shape).transpose(2, 3, 1, 0).reshape(-1, shape[0])
                first_fc = False
            else:
                model[name + '/weight'] = params[0].data.transpose(1, 0)
            if len(params) > 1:
                model[name + '/bias'] = params[1].data
        elif name.startswith('conv') or name.startswith('res'):
            if first_conv:
                model[name + '/weight'] = params[0].data[:, ::-1].transpose(2, 3, 1, 0)
                first_conv = False
            else:
                model[name + '/weight'] = params[0].data.transpose(2, 3, 1, 0)
            if len(params) > 1:
                model[name + '/bias'] = params[1].data
        elif name.startswith('bn'):
            model[name + '/running_mean'] = params[0].data
            model[name + '/running_var'] = params[1].data
            # Unknown params[2].data
        elif name.startswith('scale'):
            model[name + '/weight'] = params[0].data
            model[name + '/bias'] = params[1].data
        else:
            print('Unknown layer: %s  %s' % (name, '  '.join([str(param.data.shape) for param in params])), file=sys.stderr)
    pickle.dump(model, open(output, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download trained Caffe model and convert it to TensorFlow model')
    parser.add_argument('--prototxt', type=str, default='', help='download Caffe prototxt to or load it from this path')
    parser.add_argument('--caffemodel', type=str, default='', help='download Caffe model to or load it from this path')
    parser.add_argument('-o', dest='output', type=str, default='', help='save TensorFlow model to this path')
    parser.add_argument('--model', type=str, choices=list(caffe_models.keys()),
                        default='alexnet', help='Model to download')
    args = parser.parse_args()
    if len(args.prototxt) == 0:
        args.prototxt = os.path.join(os.path.dirname(__file__), caffe_models[args.model][3])
    if len(args.caffemodel) == 0:
        args.caffemodel = os.path.join(os.path.dirname(__file__), caffe_models[args.model][4])
    if len(args.output) == 0:
        args.output = os.path.join(os.path.dirname(__file__), caffe_models[args.model][5])

    if not os.path.exists(args.prototxt):
        print('Downloading prototxt...')
        download(args.prototxt, caffe_models[args.model][0])

    if not os.path.exists(args.caffemodel):
        print('Downloading model...')
        download(args.caffemodel, caffe_models[args.model][1])
        if not check(args.caffemodel, sha1=caffe_models[args.model][2]):
            print('ERROR: model did not download correctly! Run this again.')
            sys.exit(1)

    print('Extracting model...')
    extract_model(args.prototxt, args.caffemodel, args.output, caffe_models[args.model][6])
