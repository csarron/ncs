from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from helper import save_text, check_file_existence
from see_net import print_net
import os

os.environ['GLOG_minloglevel'] = '2'

try:
    import caffe
except Exception as e:
    print(e)


# todo: add visualization functionality
# def visualize(net)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_file", type=check_file_existence,
                        help="caffe model file (.prototxt)")
    parser.add_argument("-w", "--weights_file", type=check_file_existence,
                        help="caffe weights file (.caffemodel)")
    parser.add_argument("-t", "--text_format", action="store_true",
                        help="whether to save weights file in text format or not")
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
