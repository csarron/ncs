from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import os
import subprocess
import glob
import json
from PIL import Image
from collections import OrderedDict


# convert caffe models to dlc files
def create_dlc(proto_path_, weights_path_=None):
    if 'SNPE_ROOT' not in os.environ:
        raise RuntimeError('SNPE_ROOT not setup.  Please run the SDK env setup script.')
    snpe_root = os.path.abspath(os.environ['SNPE_ROOT'])
    if not os.path.isdir(snpe_root):
        raise RuntimeError('SNPE_ROOT (%s) is not a dir' % snpe_root)
    if not os.path.exists(proto_path_):
        raise RuntimeError('%s does not exist!' % proto_path_)

    if not weights_path_:
        weights_path_ = proto_path_.replace('.prototxt', '.caffemodel')
    if not os.path.exists(weights_path_):
        raise RuntimeError('%s does not exist!' % weights_path_)
    print('Creating DLC')
    dlc_path = proto_path_.replace('.prototxt', '.dlc')
    cmd_ = ['snpe-caffe-to-dlc',
            '--caffe_txt', proto_path_,
            '--caffe_bin', weights_path_,
            '--dlc', dlc_path]
    subprocess.call(cmd_)
    return dlc_path


def crop_image(img_path, cropped_path, target_size):
    img = Image.open(img_path)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(list(np.shape(img))) == 2:
        img = img.convert(mode='RGB')
    # center crop to square
    width, height = img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    img = img.crop(crop_coord)
    # resize to target size
    img = img.resize((target_size, target_size), Image.ANTIALIAS)
    # save output
    img.save(cropped_path)
    return cropped_path


def create_mean_npy(mean_file_, img_size):
    img_mean_npy = np.load(mean_file_)  # should be (3, 256, 256)
    # crop to 3, img_size, img_size that is smaller - usable by caffe
    mean_npy = np.ndarray(shape=(3, img_size, img_size))

    if img_mean_npy.shape[0] < mean_npy.shape[0] \
            or img_mean_npy.shape[1] < mean_npy.shape[1] \
            or img_mean_npy.shape[2] < mean_npy.shape[2]:
        raise RuntimeError(
            'Bad mean shape {} for image mean shape {}'.format(img_mean_npy.shape, mean_npy.shape))
    # cut to size
    mean_npy = img_mean_npy[:, :img_size, :img_size]
    # transpose to 227, 227, 3 for snpe mean subtraction
    return np.transpose(mean_npy, (1, 2, 0))


def img_to_snpe_raw(img_file_, mean_npy, img_size):
    img = Image.open(img_file_)
    img_array = np.array(img)  # read it

    # reshape to target size
    img_ndarray = np.reshape(img_array, (img_size, img_size, 3))
    # reverse last dimension: rgb -> bgr
    img_out = img_ndarray[..., ::-1]
    # mean subtract
    img_out = img_out - mean_npy
    img_out = img_out.astype(np.float32)
    # save
    fid = open(img_file_ + '.raw', 'wb')
    img_out.tofile(fid)
    return img_file_ + '.raw'


def create_file_list(input_dir, output_filename, ext_pattern, print_out=False, rel_path=False):
    input_dir = os.path.abspath(input_dir)
    output_filename = os.path.abspath(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.isdir(input_dir):
        raise RuntimeError('input_dir %s is not a directory' % input_dir)

    if not os.path.isdir(output_dir):
        raise RuntimeError('output_filename %s directory does not exist' % output_dir)

    glob_path = os.path.join(input_dir, ext_pattern)
    file_list = glob.glob(glob_path)

    if rel_path:
        file_list = [os.path.relpath(file_path, output_dir) for file_path in file_list]

    if len(file_list) <= 0:
        if print_out:
            print('No results with %s' % glob_path)
    else:
        with open(output_filename, 'w') as f:
            f.write('\n'.join(file_list))
            if print_out:
                print('%s created listing %d files.' % (output_filename, len(file_list)))


# create input imagenet
def create_inputs(img_size=227):
    print('Create SNPE network input')
    scripts_dir = os.getcwd()
    image_dir = os.path.join(scripts_dir, 'image_data')
    mean_npy = os.path.join(image_dir, 'ilsvrc_2012_mean.npy')
    data_cropped_dir = os.path.join(image_dir, 'crop_%s' % img_size)
    if not os.path.exists(data_cropped_dir):
        os.makedirs(data_cropped_dir)

    for img_file in glob.glob(os.path.join(image_dir, '*.jpg')):
        cropped_img_path = os.path.join(data_cropped_dir, os.path.basename(img_file))
        print(cropped_img_path)
        if not os.path.exists(cropped_img_path):
            crop_image(img_file, cropped_img_path, img_size)
            image_mean = create_mean_npy(mean_npy, img_size)
            img_to_snpe_raw(cropped_img_path, image_mean, img_size)

    print('Create file lists')
    raw_list = os.path.join(image_dir, 'target_raw_list_%s.txt' % img_size)
    if not os.path.exists(raw_list):
        create_file_list(data_cropped_dir, raw_list, '*.raw', print_out=True, rel_path=False)
    return data_cropped_dir, raw_list


# generate bench config json file
def gen_config(dlc_path, input_list_file, input_data):
    name = os.path.splitext(os.path.basename(dlc_path))[0]
    config = OrderedDict()
    config['Name'] = name
    config['HostRootPath'] = name
    config['HostResultsDir'] = os.path.join(name, 'results')
    config['DevicePath'] = '/data/local/tmp/snpebm'
    config['Devices'] = ["1234"]
    config['Runs'] = 2

    model = OrderedDict()
    model['Name'] = name
    model['Dlc'] = dlc_path
    model['InputList'] = input_list_file
    model['Data'] = [input_data]
    config['Model'] = model

    config['Runtimes'] = ['GPU', 'CPU']
    config['Measurements'] = ['timing', 'mem']

    json_path = name + '.json'
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=4)
    return json_path


# run snpe bench script
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--proto_file", type=str,
                        help="caffe proto file (.prototxt)")
    parser.add_argument("-s", "--image_size", type=int, default=227,
                        help="input image size")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dlc_file = create_dlc(args.proto_file)
    data_dir, raw_file_list = create_inputs(args.image_size)
    config_path = gen_config(dlc_file, raw_file_list, data_dir)
    bench_cmd = ['python', 'snpe_bench.py', '-c', config_path, '-a']
    subprocess.call(bench_cmd)
