# NCS

Caffe Model Tools for Intel Movidius Neural Compute Stick

## Usage

### 1. Engergy Measurement (Monsoon Power Monitor)

The `pt4reader.py` script under energy folder can be used to parse Monsoon binary `.pt4` file. This script has no python dependencies, it should work as long as there is a Python environment.

use `python pt4reader.py sample.pt4` to print the values in terminal/console, or 
use `python pt4reader.py sample.pt4 > sample.csv` to save to a `csv` file.


### 2. Model Generation

Note PyCaffe should be installed in the current Python environment

#### Generate Weights

The `gen_weights.py` script can be used to create a model's weights file (`.caffemodel`) 
given the model's deployment/definition file (`.prototxt`)

use `python gen_weights.py model.prototxt`

#### AlexNet style networks

1. edit the searching parameters in `nets/alexnet/alexnet.yaml` or just use the default one
2. run `./gen.sh nets/alexnet/alexnet.yaml 10000`
(note the `gen.sh` script will call `gen_net.py`, 10000 means the try time since not every run will generate valid Caffe model)

#### Networks in the paper “[Convolutional Neural Networks at Constrained Time Cost](https://arxiv.org/abs/1412.1710)”


#### ResNets

`cd pynetbuilder/`
example net arch generation:

`-s` means save weights
other args same as original code

`python app/imagenet/build_resnet.py -m normal -b 3 4 6 3  -n 32 --no-fc_layers -f resnet34-1x32d -s`
`python app/imagenet/build_resnet.py -m bottleneck -b 3 8 36 3 -n 128 --no-fc_layers -f resnet152_1x32d -s`
`python app/imagenet/build_resnet.py -m bottleneck -b 3 30 48 8 -n 32 --no-fc_layers -f resnet269_1 -s`
`python app/imagenet/build_resnet.py -m normal -b 2 2 2 2 -n 256 --no-fc_layers -f wrn18-3 -s`


### 3. Model Analysis

Note PyCaffe should be installed in the current Python environment

#### Model Inspection


#### Model Profiling


