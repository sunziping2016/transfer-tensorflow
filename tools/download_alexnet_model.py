import os
import sys
import argparse
import time
import hashlib
from six.moves import urllib

caffemodel_prototxt = 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt'
caffemodel_url = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
caffemodel_sha1 = '9116a64c0fbe4459d18f4bb6b56d647b63920377'


def download(filename, url):
    def reporthook(count, block_size, total_size):
        """
        From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
        """
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = (time.time() - start_time) or 0.01
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                         (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    urllib.request.urlretrieve(url, filename, reporthook)


def check(filename, sha1):
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1


def parameter_provider(model, name, params):
    if name.startswith('fc'):
        model[name + '/weights'] = params[0].data.transpose(1, 0)
        if len(params) > 1:
            model[name + '/biases'] = params[1].data
    elif name.startswith('conv'):
        model[name + '/weights'] = params[0].data.transpose(2, 3, 1, 0)
        if len(params) > 1:
            model[name + '/biases'] = params[1].data


def extract_model(prototxt, model, output):
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
    import pickle
    net = caffe.Net(prototxt, model, caffe.TEST)
    model = {}
    for layer in net.params:
        parameter_provider(model, layer, net.params[layer])
    pickle.dump(model, open(output, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download trained Caffe Alexnet and convert it to TensorFlow model')
    parser.add_argument('--prototxt', type=str, default='deploy.prototxt',
                        help='download caffe prototxt to or load it from this path')
    parser.add_argument('--model', type=str, default='bvlc_alexnet.caffemodel',
                        help='download caffe model to or load it from this path')
    parser.add_argument('-o', dest='output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '../models/alexnet.pkl'),
                        help='save tensorflow model to this path')
    args = parser.parse_args()

    if not os.path.exists(args.prototxt):
        print('Downloading prototxt...')
        download(args.prototxt, caffemodel_prototxt)

    if not os.path.exists(args.model):
        print('Downloading model...')
        download(args.model, caffemodel_url)
        if not check(args.model, sha1=caffemodel_sha1):
            print('ERROR: model did not download correctly! Run this again.')
            sys.exit(1)

    print('Extracting model...')
    extract_model(args.prototxt, args.model, args.output)
