from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import enum
import os
import sys

from helper import check_file_existence

os.environ['GLOG_minloglevel'] = '2'

try:
    import caffe
except Exception as e:
    print(e)


class Layer(enum.Enum):
    DATA = 'data'
    INPUT = 'input'
    CONV = 'conv'
    POOL = 'pool'
    FC = 'fc'
    OUTPUT = 'output'


def save_net_spec(model_spec_, model_path_, model_name_):
    model_proto = model_spec_.to_proto()
    model_proto.name = model_name_
    proto_file = os.path.join(model_path_, '{}.prototxt'.format(model_name_))
    with open(proto_file, 'w') as f:
        f.write(str(model_proto))


def parse_param(param_file_):
    import yaml
    with open(param_file_, 'r') as f:
        net_params_ = yaml.load(f)
        # print(net_params_)
    return net_params_


def create_net_spec(net_params_):
    net_spec_ = caffe.NetSpec()
    ''' net_params_ example
    net_params_ = [0: ['data', [1, 1, 28, 28]],
               1: ['conv', 7, 2, 32],
               2: ['pool', 3, 2],
               3: ['conv', 5, 1, 32],
               4: ['pool', 2, 1],
               5: ['fc', 512],
               6: ['relu', ''],
               7: ['output', 10]]
    '''

    layer = caffe.layers.Input(shape=[dict(dim=net_params_['input'])])
    setattr(net_spec_, Layer.DATA.value, layer)
    previous_layer_name = Layer.DATA.value
    layer_index = 1
    for layer_spec in net_params_['spec']:
        layer_name = layer_spec['layer']
        layer_type = Layer(layer_name)

        num = layer_spec.get('num', 0)
        kernel = layer_spec.get('kernel', 0)

        if isinstance(kernel, list):
            k_h = kernel[0]
            k_w = kernel[1]
        else:
            k_h = k_w = kernel

        stride = layer_spec.get('stride', 1)
        if isinstance(stride, list):
            s_h = stride[0]
            s_w = stride[1]
        else:
            s_h = s_w = stride

        pad = layer_spec.get('pad', 0)
        if isinstance(pad, list):
            p_h = pad[0]
            p_w = pad[1]
        else:
            p_h = p_w = pad

        repeat = layer_spec.get('repeat', 1)

        if layer_type is Layer.CONV:
            for i in range(repeat):
                previous_layer = getattr(net_spec_, previous_layer_name)
                layer = caffe.layers.Convolution(previous_layer, num_output=num,
                                                 kernel_h=k_h, kernel_w=k_w,
                                                 stride_h=s_h, stride_w=s_w,
                                                 pad_h=p_h, pad_w=p_w,
                                                 weight_filler=dict(type='xavier'),
                                                 bias_filler=dict(type='constant'))
                setattr(net_spec_, layer_name + str(layer_index), layer)
                layer_index += 1

                # attach relu
                layer = caffe.layers.ReLU(layer, in_place=True)
                setattr(net_spec_, 'relu' + str(layer_index), layer)
                previous_layer_name = 'relu' + str(layer_index)
                layer_index += 1

        elif layer_type is Layer.POOL:
            previous_layer = getattr(net_spec_, previous_layer_name)
            layer = caffe.layers.Pooling(previous_layer, pool=caffe.params.Pooling.MAX,
                                         kernel_h=k_h, kernel_w=k_w,
                                         stride_h=s_h, stride_w=s_w,
                                         pad_h=p_h, pad_w=p_w)
            setattr(net_spec_, layer_name + str(layer_index), layer)
            previous_layer_name = layer_name + str(layer_index)
            layer_index += 1

        elif layer_type == Layer.FC:
            for i in range(repeat):
                previous_layer = getattr(net_spec_, previous_layer_name)

                layer = caffe.layers.InnerProduct(previous_layer, num_output=num,
                                                  weight_filler=dict(type='xavier'),
                                                  bias_filler=dict(type='constant'))
                setattr(net_spec_, layer_name + str(layer_index), layer)
                layer_index += 1

                # attach relu
                layer = caffe.layers.ReLU(layer, in_place=True)
                setattr(net_spec_, 'relu' + str(layer_index), layer)
                previous_layer_name = 'relu' + str(layer_index)
                layer_index += 1

        else:
            raise ValueError('unsupported layer:{}'.format(layer_name))

    previous_layer = getattr(net_spec_, previous_layer_name)
    layer = caffe.layers.InnerProduct(previous_layer, num_output=net_params_['output'],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant'))
    setattr(net_spec_, Layer.OUTPUT.value, layer)
    return net_spec_


def create_net(param_file_):
    net_params = parse_param(param_file_)

    net_path_prefix = os.path.splitext(param_file)[0]
    model_name = os.path.basename(net_path_prefix)

    model_folder = os.path.join(os.path.dirname(param_file_), 'gen')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    net_spec = create_net_spec(net_params)
    save_net_spec(net_spec, model_folder, model_name)
    proto_file = os.path.join(model_folder, '{}.prototxt'.format(model_name))
    from gen_weights import save_net_weights
    save_net_weights(proto_file)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: enter the path of caffe model proto file (.prototxt)")
        sys.exit()

    param_file = sys.argv[1]
    check_file_existence(param_file)
    if not param_file:
        raise ValueError('please use -p to specify model parameters!')

    create_net(param_file)
