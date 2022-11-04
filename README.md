# NN-Inference-Optimization
Scripts from Neural network inference on Pytorch with tools like ONNX, TensorRT, nvFuser, TorchDynamo

## Requirements and Setup

Python 3.8 is recommended (that is where I run the code on conda environment)
For measuring GPU performance, one needs to make sure to synchronize between CPU/GPU before measuring how much time a function takes to execute. Therefore, we run few warm-up iterations to get an appromixate measure on speedups across different tools and batch sizes.

Since this benchmark includes lots of packages from really 'problematic' repos (lol), you might have difficulties setting up the environment, especially dependencies related to Nvidia tools. It is not recommended to follow instructions on blogs, or reddit. They are almost always outdated. Try to follow guides (especially Nvidia ones on the downloads page only. They are most of the time up-to-date.
Either install all the packages (CUDA >= 11.4.x, cuDNN 8.2, TensorRT 8.2.1) one by one, or through docker from [here](https://nvidia.github.io/nvidia-docker/).

The following command installs GPU PyTorch+TorchDynamo along with GPU TorchDynamo dependencies (for CUDA 11.7):

`pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117`

Some onnx and nvidia packages needs to be installed independently.
```
pip install onnx
pip install nvidia-pyindex
pip install nvidia-tensorrt
pip install onnxoptimizer
pip install nvidia-tensorrt
```

## Benchmark setup

Vanilla BERT (tested on randomly generated input data). Benchmarks should be pretty close for many of the transformer architectures.

## Few notes

Here is few key conclusions that I came up to when doing research towards inference optimization tools for transformers:
- As always, installing tools right takes more time than actually using those (especially Nvidia ones)
- Loads of optimization papers (more than new architectures) emerge almost every month, so that optimization can become a field on its own :sweat_smile:
Analyzing how tools like ONNX, TensorRT, etc.,works, most of them apply either one or more combinations of several methods below:
- Remove redundant operations, e.g. dropout in inference
- Perform constant folding, e.g. loading constants on compile time rather than runtime
- Kernel fusion, e.g. keeping intermediate values in cache, rather than going back n forth with RAM
- Layer fusion, e.g. compressing 2 or more compatible layers (Conv2d, ReLu, Bias..) into one
- Mixed precision/Quantization - reducing precision on weights & activations wherever possible (with loss-scaling to preserve small gradients)
- Unstructured and structured pruning - removing some neuron from non-fragile* architectures based on magnitude, movement, and structure
Brief Results in few points:
- Pytorch FP16 - 1.5x speedup across different batch sizes best being batch size 256 - largest
- ONNX FP16 - 2-3x speedup across different batch sizes, relatively same performance on all batch sizes from 1 to 256
- TensorRT - 5-8x speedup across different batch sizes, best being batch size 256 - largest
- TorchDynamo with nvFuser/nnc - 2-3x speedup across different batch sizes best being batch size 1. Unlike tools above, no conversion to intermediate format is required.

For more. Reference to the actual torchdynamo [repo](https://github.com/pytorch/torchdynamo) 
