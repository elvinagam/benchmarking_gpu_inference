# NN-Inference-Optimization
Scripts from Neural network inference on Pytorch with tools like ONNX, TensorRT, nvFuser, TorchDynamo

## Requirements and Setup

Python 3.8 is recommended (that is where I run the code on conda environment)

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
