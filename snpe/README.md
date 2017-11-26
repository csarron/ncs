# SNPE benchmark batch run script


## Features
- custom input image cropping size (default 227)
- auto convert caffe model to dlc files
- auto generate benchmarking configuration json file

## Usage

- step1: setup SNPE environment (skip this if you already set up)
    1. use bash and set CAFFE_DIR, e.g.: `CAFFE_DIR=~/caffe_snpe`
    2. `export PYTHONPATH=$CAFFE_DIR/python:$PYTHONPATH`
    3.  `source bin/envsetup.sh -c $CAFFE_DIR`
    
- step2: copy `image_data` to `$SNPE_ROOT/benchmarks` and `cd $SNPE_ROOT/benchmarks`, then
`python run_bench.py -p <path to caffe prototxt file>`