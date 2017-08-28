from .caffe_alexnet import alexnet

# Map from model name to (constructor, output feature numbers of fc)
base_models = {
    'alexnet': (alexnet, (9216, 4096, 4096, 1000))
}
