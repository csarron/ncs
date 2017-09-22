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


def parse_net_def_layers(net_proto_file_):
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format
    net_def = caffe_pb2.NetParameter()
    text_format.Merge(open(net_proto_file_).read(), net_def)
    #     print(net_def.layer) # print whole prototxt
    if net_def.layers:
        layers = net_def.layers
        # print('using layers')
    else:
        layers = net_def.layer
        # print('using layer')

    layer_names = map(lambda layer: layer.name, layers)
    layers_dict = dict(zip(layer_names, layers))
    return layers_dict


def vis_net(proto_file_, model_file_):
    import caffe
    from caffe.proto import caffe_pb2
    net = caffe.Net(proto_file_, caffe.TEST, weights=model_file_)
    layers_def_dict = parse_net_def_layers(proto_file_)

    print("{:15s}: {:15s}{:20s} ({}, {}, {}, {})\n".format('Name', 'Layer', '(n, c, h, w)',
                                                                    's_w', 's_h', 'p_w', 'p_h'))
    for name, layer in zip(net._layer_names, net.layers):
        print("{:15s}: {:15s}".format(prettify_name(name), layer.type), end='')
        layer_def = layers_def_dict.get(name, '')
        if layer_def == '':
            print()
            continue
        if layer_def.type == "Convolution" or layer_def.type == caffe_pb2.V1LayerParameter.CONVOLUTION:  # 4
            stride = layer_def.convolution_param.stride[0] if len(layer_def.convolution_param.stride) else 1
            s_h = layer_def.convolution_param.stride_h if layer_def.convolution_param.stride_h else stride
            s_w = layer_def.convolution_param.stride_w if layer_def.convolution_param.stride_w else stride
            pad = layer_def.convolution_param.pad[0] if len(layer_def.convolution_param.pad) else 0
            p_h = layer_def.convolution_param.pad_h if layer_def.convolution_param.pad_h else pad
            p_w = layer_def.convolution_param.pad_w if layer_def.convolution_param.pad_w else pad
            data_shape = net.params[name][0].data.shape
            if len(data_shape) > 1:  # ignore bias shape
                print("{:20s} ({}, {}, {}, {})".format(str(data_shape), s_w, s_h, p_w, p_h))
            else:
                print()
        elif layer_def.type == "Pooling" or layer_def.type == caffe_pb2.V1LayerParameter.POOLING:  # 17
            stride = layer_def.pooling_param.stride
            s_h = layer_def.pooling_param.stride_h if layer_def.pooling_param.stride_h else stride
            s_w = layer_def.pooling_param.stride_w if layer_def.pooling_param.stride_w else stride
            pad = layer_def.pooling_param.pad
            p_h = layer_def.convolution_param.pad_h if layer_def.convolution_param.pad_h else pad
            p_w = layer_def.convolution_param.pad_w if layer_def.convolution_param.pad_w else pad
            kernel = layer_def.pooling_param.kernel_size
            k_h = layer_def.pooling_param.kernel_h if layer_def.pooling_param.kernel_h else kernel
            k_w = layer_def.pooling_param.kernel_w if layer_def.pooling_param.kernel_w else kernel

            pool_type = caffe_pb2.PoolingParameter.PoolMethod.Name(layer_def.pooling_param.pool)
            print("{:20s} ({}, {}, {}, {})".format('(type: {}, {}, {})'.format(pool_type, k_h, k_w),
                                                   s_w, s_h, p_w, p_h))
        elif layer_def.type == "InnerProduct" or layer_def.type == caffe_pb2.V1LayerParameter.INNER_PRODUCT:  # 14
            data_shape = net.params[name][0].data.shape
            if len(data_shape) > 1:  # ignore bias shape
                print("{:20s}".format(str(data_shape)))
            else:
                print()
        else:
            print()

    print("\n\n{:15s}:  {}\n".format('FeatureMaps', '(b, c, h, w)'))
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
