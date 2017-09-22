from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys


def prettify_name(name_):
    name_str = str(name_)
    if len(name_str) > 15:
        name_show = name_str[0:10] + name_str[-5:]
    else:
        name_show = name_str
    return name_show


def vis_net(proto_file_, model_file_):
    import caffe
    net = caffe.Net(proto_file_, caffe.TEST, weights=model_file_)
    print("{:15s}: {:15s}  {:20s} \n".format("Name", "Layer", "(n, c, h, w)"))
    for name, layer in zip(net._layer_names, net.layers):
        print("{:15s}: {:15s}".format(prettify_name(name), str(layer.type)), end='')
        if len(layer.blobs):
            data_shape = net.params[name][0].data.shape
            if len(data_shape) > 1:  # ignore bias shape
                print("{:20s}".format(str(data_shape)))
            else:
                print()
        else:
            print()

    print("\nBlobs:")
    for name, blob in net.blobs.items():
        print("{:15s}:  {}".format(prettify_name(name), str(blob.data.shape)))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        proto_file = sys.argv[1]
        model_file = proto_file.replace('.prototxt', '.caffemodel')
    elif len(sys.argv) == 3:
        proto_file = sys.argv[1]
        model_file = sys.argv[2]
    else:
        print("Usage: (net.prototxt) [optional net.caffemodel]")
        sys.exit()

    proto_file = sys.argv[1]

    vis_net(proto_file, model_file)
