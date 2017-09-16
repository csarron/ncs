import argparse
import os


def check_file_existence(file_):
    if not os.path.exists(file_):
        raise argparse.ArgumentTypeError("%s does not exist!" % file_)
    # print("using graph file:", graph_file_)
    return file_


def save_text(weights_file):
    from caffe.proto import caffe_pb2
    net_def = caffe_pb2.NetParameter()

    net_def.ParseFromString(open(weights_file, 'rb').read())
    proto_text_file = '{}.txt'.format(weights_file)
    with open(proto_text_file, "w") as f:
        f.write(str(net_def))
