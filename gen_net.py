from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from helper import check_file_existence, save_text
from vis_net import print_net

try:
    import caffe
except Exception as e:
    print(e)


def create_net_spec(net_params, save_path, batch_size=1):
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


def create_solver(solver_config_path, net_def_path):
    from caffe.proto import caffe_pb2
    s = caffe_pb2.SolverParameter()
    s.train_net = net_def_path
    with open(solver_config_path, 'w') as f:
        f.write(str(s))


def save_model_weights(solver_config_path, weights_file_):
    caffe.set_mode_cpu()
    solver = caffe.SGDSolver(solver_config_path)
    solver.net.save(weights_file_)


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

    print_net(model_file, weights_file)

    if args.text_format:
        print("saving weights to text file: {}.txt".format(weights_file))
        save_text(weights_file)
        print("finished.")
