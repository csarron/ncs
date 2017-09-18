from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import enum
import itertools
import os
import numpy as np
import tempfile

from helper import check_file_existence, merge_dicts

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
    RELU = 'relu'
    FC = 'fc'
    OUTPUT = 'output'


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
    for layer_index, layer_spec in sorted(net_params_.items()):
        layer_name = layer_spec[0]
        layer_param = layer_spec[1:]
        if layer_name == Layer.DATA.value:
            layer = caffe.layers.Input(shape=[dict(dim=layer_param[0])])
            setattr(net_spec_, layer_name, layer)
            last_layer_name = layer_name
        elif layer_name == Layer.CONV.value:
            k, s, num = layer_param
            last_layer = getattr(net_spec_, last_layer_name)
            layer = caffe.layers.Convolution(last_layer, kernel_h=k, kernel_w=k,
                                             stride_h=s, stride_w=s, num_output=num,
                                             weight_filler=dict(type='xavier'),
                                             bias_filler=dict(type='constant'))
            setattr(net_spec_, layer_name + str(layer_index), layer)
            last_layer_name = layer_name + str(layer_index)
        elif layer_name == Layer.POOL.value:
            last_layer = getattr(net_spec_, last_layer_name)
            k, s = layer_param
            layer = caffe.layers.Pooling(last_layer, kernel_h=k, kernel_w=k,
                                         stride_h=s, stride_w=s,
                                         pool=caffe.params.Pooling.MAX)
            setattr(net_spec_, layer_name + str(layer_index), layer)
            last_layer_name = layer_name + str(layer_index)
        elif layer_name == Layer.RELU.value:
            last_layer = getattr(net_spec_, last_layer_name)
            layer = caffe.layers.ReLU(last_layer, in_place=True)
            setattr(net_spec_, layer_name + str(layer_index), layer)
            last_layer_name = layer_name + str(layer_index)
        elif layer_name == Layer.FC.value:
            last_layer = getattr(net_spec_, last_layer_name)
            layer = caffe.layers.InnerProduct(last_layer, num_output=layer_param[0],
                                              weight_filler=dict(type='xavier'),
                                              bias_filler=dict(type='constant'))
            setattr(net_spec_, layer_name + str(layer_index), layer)
            last_layer_name = layer_name + str(layer_index)
        elif layer_name == Layer.OUTPUT.value:
            last_layer = getattr(net_spec_, last_layer_name)

            layer = caffe.layers.InnerProduct(last_layer, num_output=layer_param[0],
                                              weight_filler=dict(type='xavier'),
                                              bias_filler=dict(type='constant'))
            setattr(net_spec_, layer_name, layer)
        else:
            raise ValueError('unsupported layer:{}'.format(layer_name))

    return net_spec_


def get_loc(layer_name, net_params):
    loc = {}
    if layer_name in net_params:
        if 'location' in net_params[layer_name]:
            locations = net_params[layer_name]['location']
            loc = dict((l, layer_name) for l in locations)
    # print(loc)
    return loc


def gen_one_net_params(net_params_):
    conv_loc = get_loc(Layer.CONV.value, net_params_)
    pool_loc = get_loc(Layer.POOL.value, net_params_)
    relu_loc = get_loc(Layer.RELU.value, net_params_)
    fc_loc = get_loc(Layer.FC.value, net_params_)
    layer_dict = merge_dicts(conv_loc, pool_loc, relu_loc, fc_loc)
    print(layer_dict)

    n_conv = len(conv_loc)
    conv_kernel_sizes = net_params_[Layer.CONV.value]['kernel_size']
    conv_strides = net_params_[Layer.CONV.value]['stride']
    conv_nums = net_params_[Layer.CONV.value]['num_output']

    n_pool = len(pool_loc)
    pool_kernel_sizes = net_params_[Layer.POOL.value]['kernel_size']
    pool_strides = net_params_[Layer.POOL.value]['stride']

    n_fc = len(fc_loc)
    fc_nums = net_params_[Layer.FC.value]['num_output']

    conv_spec = list(itertools.product(conv_kernel_sizes, conv_strides, conv_nums))
    conv_specs = list(itertools.product(*((conv_spec,) * n_conv)))
    conv_all_specs = []
    for spec in conv_specs:
        temp_spec = []
        for layer_index, spec_item in zip(conv_loc.keys(), spec):
            spec_item = (layer_index, *spec_item)
            temp_spec.append(spec_item)
        conv_all_specs.append(tuple(temp_spec))

    pool_spec = list(itertools.product(pool_kernel_sizes, pool_strides))
    pool_specs = list(itertools.product(*((pool_spec,) * n_pool)))
    pool_all_specs = []
    for spec in pool_specs:
        temp_spec = []
        for layer_index, spec_item in zip(pool_loc.keys(), spec):
            spec_item = (layer_index, *spec_item)
            temp_spec.append(spec_item)
        pool_all_specs.append(tuple(temp_spec))

    relu_specs = list(itertools.product(*([''],) * len(relu_loc)))
    relu_all_specs = []
    for spec in relu_specs:
        temp_spec = []
        for layer_index, spec_item in zip(relu_loc.keys(), spec):
            spec_item = (layer_index, spec_item)
            temp_spec.append(spec_item)
        relu_all_specs.append(tuple(temp_spec))

    fc_specs = list(itertools.product(*((fc_nums,) * n_fc)))
    fc_all_specs = []
    for spec in fc_specs:
        temp_spec = []
        for layer_index, spec_item in zip(fc_loc.keys(), spec):
            spec_item = (layer_index, spec_item)
            temp_spec.append(spec_item)
        fc_all_specs.append(tuple(temp_spec))

    all_specs = itertools.product(conv_all_specs, pool_all_specs, relu_all_specs, fc_all_specs)
    for i, one_spec in enumerate(all_specs):
        layer_specs = []
        for spec in one_spec:
            for spec_i in spec:
                layer_specs.append(spec_i)
        print(i, layer_specs)

        def f(layer_spec_):
            specs_ = list(layer_spec_)
            specs_[0] = layer_dict[specs_[0]]
            return specs_
        layer_indices = map(lambda s: s[0], layer_specs)
        layer_params = map(lambda s: f(s), layer_specs)

        net_param_dict = dict(zip(layer_indices, layer_params))
        net_param_dict[0] = [Layer.DATA.value, net_params_[Layer.INPUT.value]]
        ''' layer spec definition
        for 'conv': (kernel_w, kernel_h, stride_w, stride_h, num_output)
        for 'pool': (kernel_w, kernel_h, stride_w, stride_h)
        for 'fc5' : num_output
        for 'relu': ''
        '''
        net_param_dict[len(layer_specs) + 1] = [Layer.OUTPUT.value, net_params_[Layer.OUTPUT.value]]
        yield i + 1, net_param_dict


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
    parser.add_argument("-p", "--param_file", type=check_file_existence, default='nets/alexnet/alexnet.yaml',
                        help="caffe model parameter file (.param.yaml)")
    parser.add_argument("-s", "--net_size", type=int, default=100,
                        help="number of networks to generate")
    parser.add_argument("--seed", action='store_true',
                        help="number of networks to generate")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    param_file = args.param_file

    if not param_file:
        raise ValueError('please use -p to specify model parameters!')

    if args.seed:
        np.random.seed(0)

    create_net(param_file)
