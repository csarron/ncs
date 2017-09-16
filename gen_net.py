from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from helper import check_file_existence, get_name, save_text
from vis_net import print_net

try:
    import caffe
except Exception as e:
    print(e)


def parse_param(param_file_):
    net_params_ = param_file_
    return net_params_


def create_net_spec(net_params_, save_path, batch_size=1):
    # todo design net params format and parse them, below shows sample net def
    net_spec_ = caffe.NetSpec()
    net_spec_.data = caffe.layers.Input(shape=[dict(dim=[batch_size, 1, 28, 28])], ntop=1)
    net_spec_.conv1 = caffe.layers.Convolution(net_spec_.data, kernel_size=5, num_output=20,
                                               weight_filler=dict(type='xavier'))
    net_spec_.pool1 = caffe.layers.Pooling(net_spec_.conv1, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net_spec_.conv2 = caffe.layers.Convolution(net_spec_.pool1, kernel_size=5, num_output=50,
                                               weight_filler=dict(type='xavier'))
    net_spec_.pool2 = caffe.layers.Pooling(net_spec_.conv2, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
    net_spec_.fc1 = caffe.layers.InnerProduct(net_spec_.pool2, num_output=500, weight_filler=dict(type='xavier'))
    net_spec_.relu1 = caffe.layers.ReLU(net_spec_.fc1, in_place=True)
    net_spec_.output = caffe.layers.InnerProduct(net_spec_.relu1, num_output=10, weight_filler=dict(type='xavier'))

    save_model_spec(net_spec_, save_path)

    return net_spec_


def save_model_spec(model_spec_, proto_file_):
    with open(proto_file_, 'w') as f:
        f.write(str(model_spec_.to_proto()))


def save_model_weights(solver_path_, net_def_path_, weights_file_):
    from caffe.proto import caffe_pb2
    s = caffe_pb2.SolverParameter()
    s.train_net = net_def_path_
    with open(solver_path_, 'w') as f:
        f.write(str(s))
    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(solver_path_)
    solver.net.save(weights_file_)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param_file", type=check_file_existence, default='nets/lenet/lenet5.param',
                        help="caffe model parameter file (.param)")
    parser.add_argument("-t", "--text_format", action="store_true",
                        help="whether to save weights file in text format or not")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    param_file = args.param_file

    if not param_file:
        raise ValueError('please use -p to specify model parameters!')

    net_name = get_name(param_file)

    solver_file = param_file.replace('.param', '.solver')
    model_file = param_file.replace('.param', '.prototxt')
    weights_file = param_file.replace('.param', '.caffemodel')

    net_params = parse_param(param_file)
    create_net_spec(net_params, model_file)
    save_model_weights(solver_file, model_file, weights_file)

    print_net(model_file, weights_file)
    if args.text_format:
        print("saving weights to text file: {}.txt".format(weights_file))
        save_text(weights_file)
        print("finished.")
