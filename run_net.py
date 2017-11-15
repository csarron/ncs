from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from mvnc import mvncapi as mvnc
from helper import check_file_existence
import time


def check_input_shape(input_shape_):
    dim_strings = input_shape_.split(",")
    try:
        dims = list(map(lambda x: int(x.strip()), dim_strings))
        print("input shape:", dims)
        return dims
    except Exception as e:
        print("format error:", e)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_file", type=check_file_existence, default='/home/pi/dev/ncsdk/examples/caffe/SqueezeNet/graph',
                        help="model graph file")
    parser.add_argument("-s", "--input_shape", type=check_input_shape, default='227,227,3',
                        help="input shape, e.g. 224,224,3")
    parser.add_argument("-i", "--iterations", type=int, default=1,
                        help="iterations to run the model")
    parser.add_argument("-m", "--mark", action="store_true",
                        help="whether to set Raspberry Pi ttl mark (for monsoon timestamp) or not")
    return parser.parse_args()


def set_marker(state, mark_, channel=8):
    if not mark_:
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        if state == 'clean':
            print('cleaning up pin states')
            GPIO.cleanup()
        else:
            print('setting pin:', channel, 'state to', state)
            GPIO.setup(channel, GPIO.OUT, initial=state)
    except Exception as e:
        print(e)


def get_input_data(input_shape_):
    import numpy as np
    return np.random.rand(*input_shape_).astype(np.float16)


def get_ncs_device():
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 1)
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    return mvnc.Device(devices[0])


def load_graph(device_, graph_file_):
    start_time = time.time()

    device_.OpenDevice()
    # opt = device_.GetDeviceOption(mvnc.DeviceOption.OPTIMISATIONLIST)
    # print('device opt:', opt)
    with open(graph_file_, mode='rb') as f:
        blob = f.read()
    graph_ = device_.AllocateGraph(blob)

    end_time = time.time()
    return graph_, round((end_time - start_time) * 1000, 2)


def load_input(iterations_=1):
    start_time = time.time()

    graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, iterations_)
    # print('iter', graph.GetGraphOption(mvnc.GraphOption.ITERATIONS))
    graph.LoadTensor(input_data, 'user_object')

    end_time = time.time()
    return round((end_time - start_time) * 1000, 2)


def run_inference(graph_):
    start_time = time.time()

    output_, user_obj = graph_.GetResult()

    end_time = time.time()
    return output_, round((end_time - start_time) * 1000, 2)


def clean_up(device_, graph_):
    start_time = time.time()

    graph_.DeallocateGraph()
    device_.CloseDevice()

    end_time = time.time()
    return round((end_time - start_time) * 1000, 2)


if __name__ == '__main__':
    args = parse_args()
    graph_file = args.graph_file
    input_shape = args.input_shape
    iterations = args.iterations
    mark = args.mark

    input_data = get_input_data(input_shape)
    device = get_ncs_device()
    graph, graph_load_time = load_graph(device, graph_file)
    print('graph_load_time:', graph_load_time, 'ms')

    data_load_time = load_input(iterations)
    print('data_load_time:', data_load_time, 'ms')

    set_marker(1, mark)
    _, inference_time = run_inference(graph)
    print('inference_time:', inference_time, 'ms')
    set_marker(0, mark)

    clean_up_time = clean_up(device, graph)
    print('clean_up_time:', clean_up_time, 'ms')
    set_marker('clean', mark)
