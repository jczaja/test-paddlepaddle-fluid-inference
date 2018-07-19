# test-paddlepaddle-fluid-inference
Test Inference FLUID High Level API of PaddlePaddle.

## Building PaddlePaddle
mkdir build; cd build
cmake -DFLUID_INSTALL_DIR=<Installation pAth for FLUID High level API inference> -DCMAKE_BUILD_TYPE=Release -DWITH_FLUID_ONLY=ON -DWITH_SWIG_PY=OFF -DWITH_PYTHON=OFF -DWITH_MKL=ON  -DWITH_GPU=OFF ../ 
make 
make inference_lib_dist

## Building inference example

mkdir build; cd build
PADDLEROOT=<Installation pAth for FLUID High level API inference> cmake -DCMAKE_BUILD_TYPE=Release ../
make

## Running Inference examples

### ResNet-50, dummy data, MKLDNN , 100 iterations
FLAGS_use_mkldnn=true ./test-paddle-fluid --modeldir=../resnet50-imagenet-predict-cpu --iterations 100

### ResNet-50, dummy data, Reference CPU , 100 iterations
./test-paddle-fluid --modeldir=../resnet50-imagenet-predict-cpu --iterations 100
