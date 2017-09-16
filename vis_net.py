from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from helper import save_text, check_file_existence

try:
    import caffe
except Exception as e:
    print(e)


def vis_net(proto_file_, model_file_):
    net = caffe.Net(proto_file_, model_file_, caffe.TEST)
    print("{:<7}: {:17s} {:<10} {:<10} {:18s} \n".format("name", "layer", "top", "bottom", "(n, c, h, w)"))
    for name, layer in zip(net._layer_names, net.layers):
        print("{:<7}: {:17s} {:<10} {:<10}".format(str(name), str(layer.type), str(net.top_names[name]),
                                                   str(net.bottom_names[name])), end='')
        if len(layer.blobs):
            print(" {:18s}".format(str(net.params[name][0].data.shape)))
        else:
            print("")
    print("Blobs:")
    for name, blob in net.blobs.items():
        print("{:<5}:  {}".format(str(name), str(blob.data.shape)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", type=check_file_existence,
                        help="caffe model file (.prototxt)")
    parser.add_argument("-w", "--weights_file", type=check_file_existence,
                        help="caffe weights file (.caffemodel)")
    parser.add_argument("-t", "--text_format", action="store_true",
                        help="whether to round weights to integers or not")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model_file = args.model_file
    weights_file = args.weights_file

    if not model_file:
        raise ValueError('please use -m to specify caffe model file!')

    if not weights_file:
        weights_file = model_file.replace('.prototxt', '.caffemodel')
        check_file_existence(weights_file)

    vis_net(model_file, weights_file)

    if args.text_format:
        print("saving weights to text file: {}.txt".format(weights_file))
        save_text(weights_file)
        print("finished.")
