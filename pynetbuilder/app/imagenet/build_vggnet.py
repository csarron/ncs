"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

from caffe.proto import caffe_pb2
import google.protobuf as pb
from caffe import layers as L
from caffe import params as P
import caffe
import os
import tempfile
import sys
from argparse import ArgumentParser

sys.path.append('netbuilder')
from lego.hybrid import ConvReLULego
from lego.base import BaseLegoFunction
from tools.complexity import get_complexity

parser = ArgumentParser(description=""" This script generates imagenet vggnet train_val.prototxt files""")
parser.add_argument('-f', '--file_name', help="""Train and Test prototxt will be generated as train.prototxt and test.prototxt""")
parser.add_argument('-n', '--num_outputs', type=int, nargs='+', help="""Number of filters in the 5 vgg16 stages""", default=[64, 128, 256, 512, 512])
parser.add_argument('-s', '--save_weights', action='store_true', help="save model weights")


def write_prototxt(file_name, num_outputs):
    netspec = caffe.NetSpec()
    use_global_stats = True
    # Data layer
    netspec.data = caffe.layers.Input(shape=[dict(dim=[1, 3, 224, 224])], ntop=1)

    last = netspec.data
    # Conv layers stages
    for stage in range(1, 6):
        blocks = 2 if stage < 3 else 3
        s = 2 if stage < 2 else 1
        for b in range(1, blocks + 1):
            name = str(stage) + '_' + str(b)

            params = dict(name=name, num_output=num_outputs[stage - 1], kernel_size=3, pad=1, stride=s)
            last = ConvReLULego(params).attach(netspec, [last])

        pool_params = dict(name='pool_' + str(stage), kernel_size=2, stride=2, pool=P.Pooling.MAX)
        last = BaseLegoFunction('Pooling', pool_params).attach(netspec, [last])

    # FC layers
    ip_params = dict(name='fc6', num_output=4096)
    fc6 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [last])
    relu_params = dict(name='relu6')
    relu6 = BaseLegoFunction('ReLU', relu_params).attach(netspec, [fc6])
    # drop_params = dict(name='drop6', dropout_param=dict(dropout_ratio=0.5))
    # drop6 = BaseLegoFunction('Dropout', drop_params).attach(netspec, [relu6])

    ip_params = dict(name='fc7', num_output=4096)
    fc7 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [relu6])
    relu_params = dict(name='relu7')
    relu7 = BaseLegoFunction('ReLU', relu_params).attach(netspec, [fc7])
    # drop_params = dict(name='drop7', dropout_param=dict(dropout_ratio=0.5))
    # drop7 = BaseLegoFunction('Dropout', drop_params).attach(netspec, [relu7])

    ip_params = dict(name='fc8', num_output=1000)
    fc8 = BaseLegoFunction('InnerProduct', ip_params).attach(netspec, [relu7])

    model_proto = netspec.to_proto()
    model_proto.name = file_name
    fp = open(file_name+'.prototxt', 'w')
    print >> fp, model_proto
    fp.close()

def save_net_weights(model_name_):
    from caffe.proto import caffe_pb2
    solver_param = caffe_pb2.SolverParameter()
    proto_file_ = '{}.prototxt'.format(model_name_)
    weights_file_ = '{}.caffemodel'.format(model_name_)
    solver_param.train_net = os.path.abspath(proto_file_)

    with tempfile.NamedTemporaryFile(delete=False) as solver_file:
        # print('solver file:', solver_file.name)
        solver_file.write(str(solver_param).encode())
        solver_file.close()
        caffe.set_mode_cpu()
        solver = caffe.SGDSolver(solver_file.name)
        solver.net.save(weights_file_)
        os.remove(solver_file.name)

if __name__ == '__main__':
    args = parser.parse_args()
    # write_prototxt(True, 'train', args.output_folder)
    write_prototxt(args.file_name, args.num_outputs)
    if args.save_weights:
        save_net_weights(args.file_name)
    # filepath = args.output_folder + '/train.prototxt'
    # params, flops = get_complexity(prototxt_file=filepath)
    # print 'Number of params: ', (1.0 * params) / 1000000.0, ' Million'
    # print 'Number of flops: ', (1.0 * flops) / 1000000.0, ' Million'

