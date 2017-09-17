from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import collections
import os
import tempfile

from helper import check_file_existence, merge_dicts

os.environ['GLOG_minloglevel'] = '2'

try:
    import caffe
except Exception as e:
    print(e)


def save_net_spec(model_spec_, model_path_, model_name_):
    model_proto = model_spec_.to_proto()
    model_proto.name = model_name_
    proto_file = os.path.join(model_path_, '{}.prototxt'.format(model_name_))
    with open(proto_file, 'w') as f:
        f.write(str(model_proto))


def save_net_weights(model_folder_, model_name_):
    from caffe.proto import caffe_pb2
    solver_param = caffe_pb2.SolverParameter()
    proto_file_ = os.path.join(model_folder_, '{}.prototxt'.format(model_name_))
    weights_file_ = os.path.join(model_folder_, '{}.caffemodel'.format(model_name_))
    solver_param.train_net = os.path.abspath(proto_file_)

    with tempfile.NamedTemporaryFile(delete=False) as solver_file:
        # print('solver file:', solver_file.name)
        solver_file.write(str(solver_param).encode())
        solver_file.close()
        caffe.set_mode_cpu()
        solver = caffe.SGDSolver(solver_file.name)
        solver.net.save(weights_file_)
        os.remove(solver_file.name)


def parse_param(param_file_):
    import yaml
    with open(param_file_, 'r') as f:
        net_params_ = yaml.load(f)
        print(net_params_)
    return net_params_


def create_net_spec(net_params_):
    net_spec_ = caffe.NetSpec()
    ''' net_params_ example
    net_params_ = [('data', [1, 1, 28, 28]),
               ('conv1', [7, 3, 1, 1, 32]),
               ('pool2', [5, 5, 2, 1]),
               ('conv3', [5, 5, 1, 1, 64]),
               ('pool4', [2, 2, 1, 1]),
               ('fc5', 512),
               ('relu6', ''),
               ('output', 10)]
    net_params_ = collections.OrderedDict(net_params_)
    '''
    last_layer_name = ''
    for layer_name, layer_spec in net_params_.items():
        if layer_name == 'data':
            layer = caffe.layers.Input(shape=[dict(dim=layer_spec)])
            setattr(net_spec_, layer_name, layer)
            last_layer_name = layer_name
        elif layer_name.startswith('conv'):
            kh, kw, sh, sw, num = layer_spec
            last_layer = getattr(net_spec_, last_layer_name)
            layer = caffe.layers.Convolution(last_layer, kernel_h=kh, kernel_w=kw,
                                             stride_h=sh, stride_w=sw, num_output=num,
                                             weight_filler=dict(type='xavier'),
                                             bias_filler=dict(type='constant'))
            setattr(net_spec_, layer_name, layer)
            last_layer_name = layer_name
        elif layer_name.startswith('pool'):
            last_layer = getattr(net_spec_, last_layer_name)
            kh, kw, sh, sw = layer_spec
            layer = caffe.layers.Pooling(last_layer, kernel_h=kh, kernel_w=kw,
                                         stride_h=sh, stride_w=sw,
                                         pool=caffe.params.Pooling.MAX)
            setattr(net_spec_, layer_name, layer)
            last_layer_name = layer_name
        elif layer_name.startswith('relu'):
            last_layer = getattr(net_spec_, last_layer_name)
            layer = caffe.layers.ReLU(last_layer, in_place=True)
            setattr(net_spec_, layer_name, layer)
            last_layer_name = layer_name
        elif layer_name.startswith('fc'):
            last_layer = getattr(net_spec_, last_layer_name)
            layer = caffe.layers.InnerProduct(last_layer, num_output=layer_spec,
                                              weight_filler=dict(type='xavier'),
                                              bias_filler=dict(type='constant'))
            setattr(net_spec_, layer_name, layer)
            last_layer_name = layer_name
        elif layer_name == 'output':
            last_layer = getattr(net_spec_, last_layer_name)

            layer = caffe.layers.InnerProduct(last_layer, num_output=layer_spec,
                                              weight_filler=dict(type='xavier'),
                                              bias_filler=dict(type='constant'))
            setattr(net_spec_, layer_name, layer)
        else:
            raise ValueError('unsupported layer:{}'.format(layer_name))

    return net_spec_


def get_loc(layer_name, net_params):
    loc = []
    if layer_name in net_params:
        if 'location' in net_params[layer_name]:
            locations = net_params[layer_name]['location']
            loc = dict((l, layer_name) for l in locations)
    # print(loc)
    return loc


def gen_one_net_params(net_params_):
    conv_loc = get_loc('conv', net_params_)
    pool_loc = get_loc('pool', net_params_)
    relu_loc = get_loc('relu', net_params_)
    fc_loc = get_loc('fc', net_params_)
    layer_dict = merge_dicts(conv_loc, pool_loc, relu_loc, fc_loc)
    print(layer_dict)

    net_param_dict = collections.OrderedDict()
    net_param_dict['data'] = net_params_['input']
    for layer_index, layer_name in sorted(layer_dict.items()):
        print(layer_index, layer_name)
        layer_spec = ''

        ''' layer spec definition
        for 'conv': [kernel_w, kernel_h, stride_w, stride_h, num_output]
        for 'pool': [kernel_w, kernel_h, stride_w, stride_h]
        for 'fc5' : num_output
        for 'relu': ''
        '''
        net_param_dict['{}{}'.format(layer_name, layer_index)] = layer_spec

    net_param_dict['output'] = net_params_['output']
    yield 1, net_param_dict


def create_net(param_file_):
    net_params = parse_param(param_file_)

    net_path_prefix = os.path.splitext(param_file)[0]
    model_base_name = os.path.basename(net_path_prefix)

    model_folder = os.path.join(os.path.dirname(param_file_), 'gen')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for model_suffix, one_net_params in gen_one_net_params(net_params):
        model_name = '{}_{}'.format(model_base_name, model_suffix)
        net_spec = create_net_spec(one_net_params)
        save_net_spec(net_spec, model_folder, model_name)
        save_net_weights(model_folder, model_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param_file", type=check_file_existence, default='nets/lenet/lenet.yaml',
                        help="caffe model parameter file (.param.yaml)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    param_file = args.param_file

    if not param_file:
        raise ValueError('please use -p to specify model parameters!')

    create_net(param_file)
