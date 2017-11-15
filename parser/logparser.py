import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ARCHITECTURE(object):
    """
    Class for parsing Neural Network Architecture
    """

    def __init__(self, datafolder, debug=True):
        self.debug = debug
        self.datafolder = datafolder

    def get_network_architectures(self):
        return glob.glob(self.datafolder + '/*.net.txt')

    def parse_net_architecture_file(self, filename):

        result = {'arch': [], 'param_sum': [], 'feature_map_sum': [], 'feature_maps': []}

        with open(filename, 'r') as fp:
            for line in fp:
                fields = line.strip().split(':')

                # Neglecting empty lines
                if len(fields) < 2:
                    continue

                parse_res, target = self.parse_net_fields(fields)

                if parse_res:
                    result[target].append(parse_res)

        return result

    def parse_net_runtime_file(self, filename):

        result = {'layer_time': []}

        with open(filename, 'r') as fp:
            for line in fp:
                fields = list(map(lambda x: x.strip(), line.strip().split('    ')))
                # print(fields)
                # print(len(fields))

                # Neglecting empty lines
                if len(fields) < 5 or fields[0].startswith('Layer'):
                    continue

                fields_fil = list(filter(lambda x: x != '', fields))
                # print(fields_fil)

                if fields[0].strip().startswith('Total'):
                    result['total_inference_time'] = float(fields_fil[-1])
                else:
                    result['layer_time'].append({fields_fil[1]: list(map(float, fields_fil[-3:]))})

        return result

    def parse_net_runtime(self, net_filename, ncore=12):
        net_data = self.parse_net_architecture_file(net_filename)
        # print(net_data)
        net_data['run_time'] = []

        core_file_available = [False] * 12

        # print(net_filename)
        min_inferece_time = np.inf
        for i in range(1, ncore + 1):
            runtime_file = '{:s}_{:d}.txt'.format(net_filename.split('.')[0], i)

            if os.path.exists(runtime_file):
                # print('Parsing file: {:s}'.format(runtime_file))
                core_file_available[i - 1] = True

                # Parse the runtime file here
                result = self.parse_net_runtime_file(runtime_file)
                # print(result)
                net_data['run_time'].append({i: result})

                if min_inferece_time > result['total_inference_time']:
                    min_inferece_time = result['total_inference_time']
                    net_data['min_inference_time'] = min_inferece_time
            else:
                # print('File: {:s} Not Found!!!'.format(runtime_file))
                pass

        if np.sum(core_file_available) == 0:
            return None
        else:
            return net_data

    @staticmethod
    def parse_net_fields(fields):

        fields = list(map(lambda x: x.strip(), fields))

        if fields[0].lower() == 'name':
            return None, None

        if fields[0].lower() == 'data':
            return None, None

        if fields[1].startswith('Convolution'):
            conv_param = fields[-1].split('    ')

            kernels = conv_param[1].strip()[1:-1]
            kernels = list(map(int, kernels.split(',')))
            # print(kernels)

            strides = conv_param[-1].strip()[1:-1]
            strides = list(map(int, strides.split(',')))
            # print(strides)

            return {fields[0]: [kernels, strides]}, 'arch'

        if fields[1].startswith('Pooling'):
            param = fields[-1].split('  ')
            # print(param)
            types = param[0][:-1].split(',')
            strides = param[-1].strip()[1:-1]
            strides = list(map(int, strides.split(',')))

            return {fields[0]: [types[0], list(map(lambda x: int(x.strip()), types[1:])), strides]}, 'arch'

        if fields[0].lower().startswith('relu'):
            return {fields[0]: ''}, 'arch'

        if fields[1].startswith('InnerProduct'):
            param = fields[-1].split('  ')
            # print(param)
            param = param[-1].strip()[1:-1]
            param = list(map(int, param.split(',')))
            return {fields[0]: param}, 'arch'

        if fields[0].lower().startswith('param_sum'):
            return int(fields[-1]), 'param_sum'

        if fields[0].lower().startswith('feature_map_sum'):
            return int(fields[-1]), 'feature_map_sum'

        if fields[0].strip() == 'FeatureMaps':
            return None, None

        # print(fields[1])
        return {fields[0]: list(map(int, fields[1][1:-1].split(',')))}, 'feature_maps'


def parse():
    AR = ARCHITECTURE(datafolder='profiles')
    arch_files = AR.get_network_architectures()
    print('Arch. file found: ', len(arch_files))

    # res = AR.parse_net_architecture(arch_files[-1])
    # print(res.keys())

    all_net_data = []

    for file_name in arch_files:
        res = AR.parse_net_runtime(file_name)
        if res:
            all_net_data.append(res)

    print('No. of valid architectures => {:d}'.format(len(all_net_data)))
    return all_net_data


def get_param_inferencetime_featuremap(net_data):
    data = []
    for res in net_data:
        data.append([res['param_sum'][0], res['feature_map_sum'][0], res['min_inference_time']])

    np.save('net_data.npy', data)

    return np.array(data)


def plot1(data_file):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)

    data = np.load(data_file)

    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')
    ax.scatter(data[:, 0] * data[:, 1], data[:, 2], c='r', marker='o')
    ax.set_xlabel('param_sum * feature_map_sum')
    # ax.set_ylabel('feature_map_sum')
    ax.set_ylabel('min_inference_time')
    # plt.savefig("res.pdf", bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    parsed_data = parse()
    data = get_param_inferencetime_featuremap(parsed_data)
    # print(data)

    plot1('net_data.npy')
