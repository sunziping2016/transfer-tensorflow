from .caffe_alexnet import *
from .caffe_mean_file import *

# Map from model name to (constructor, output feature numbers of fc, mean file)
base_models = {
    'alexnet': (alexnet, (9216, 4096, 4096, 1000))
}
