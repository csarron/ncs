from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tempfile

from helper import check_file_existence

os.environ['GLOG_minloglevel'] = '2'

try:
    import caffe
except Exception as e:
    print(e)


def save_net_weights(proto_file_):
    from caffe.proto import caffe_pb2
    solver_param = caffe_pb2.SolverParameter()
    weights_file_ = proto_file_.replace('.prototxt', '.caffemodel')
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

    if len(sys.argv) != 2:
        print("Usage: enter the path of caffe model proto file (.prototxt)")
        sys.exit()

    proto_file = sys.argv[1]
    check_file_existence(proto_file)
    save_net_weights(proto_file)
