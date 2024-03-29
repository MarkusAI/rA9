# rA9
<p align="center">
  <img src=https://user-images.githubusercontent.com/42883224/100475287-11acfd00-3126-11eb-8f17-ae8d230a999f.png>
</p>
<hr>

Spiking Neural Network Library Based on [JAX](https://github.com/google/jax) and referencing codes from [bintorch](https://github.com/bingo619/bintorch)

The learning algorithm of this library is spike-based backpropagation proposed from [Enabling Spike-Based Backpropagation for Training Deep Neural Network Architectures](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7059737/)

## Compatitable Operating Systems

Only supports Linux and MacOS, because of the dependency of JAX and they try to [fix](https://github.com/google/jax/pull/4843) it.
But you can run this library in [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10)

## Installation

### MacOS

#### CPU
Simple, just type 
> pip install git+https://github.com/MarkusAI/rA9
#### GPU
MacOS does not support NVIDIA CUDA. 
### Linux

#### CPU
Simple, just type 
> pip install git+https://github.com/MarkusAI/rA9
#### GPU
You need to setup JAX before installing the rA9 as [GPU-dedicated](https://github.com/google/jax#pip-installation)
and install rA9 as following:
> pip install git+https://github.com/MarkusAI/rA9

### Example
 
Check LeNet.py in example

