from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np

try:
    import caffe
except Exception as e:
    print(e)


def round_weights(net, weights_file_):
    for name, layer in zip(net._layer_names, net.layers):
        if len(layer.blobs):
            layer_data = net.params[name]
            layer_weights = layer_data[0].data
            print("{:10s}: {:8s}".format(name, str(layer_weights.shape)), end='')
            layer_weights.flat = np.rint(layer_weights.flat).astype(int).flatten()
            if len(layer_data) == 2:
                layer_biases = layer_data[1].data
                print(",{:8s}".format(str(layer_biases.shape)))
                layer_biases.flat = np.rint(layer_biases.flat).astype(int).flatten()
            else:
                print()
    net.save(weights_file_)


def prune_weights(net, threshold=0.001):
    pass


def factorize_weights(net, retain_ratio):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proto_file", type=str,
                        default='net_configs/create_net_spec/create_net_spec.prototxt',
                        help="caffe proto file (.prototxt)")
    parser.add_argument("-m", "--model_file", type=str, default='',
                        help="caffe model file (.caffemodel)")
    parser.add_argument("-r", "--round_weights", action="store_true",
                        help="whether to round weights to integers or not")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    proto_file = args.proto_file
    model_file = args.model_file
    if not model_file:
        model_file = proto_file.replace('.prototxt', '.caffemodel')

    caffe_net = caffe.Net(proto_file, caffe.TEST, weights=model_file)
    if args.round_weights:
        print("rounding weights to integers...")
        round_weights(caffe_net, model_file.replace('.caffemodel', '.rint.caffemodel'))
        print("finished.")
