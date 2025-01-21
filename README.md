# TinyTorch

A toy framework for deep learning. My homework for PKU undergraduate course "Programming in AI" 2024 fall.

Inspired by [interestingLSY/NeuroFrame - GitHub](https://github.com/interestingLSY/NeuroFrame/tree/master).

## Installation

Run `python setup.py develop` to build the tinytorch module.

## Structure

The `src` folder contains all the source C++ code of `TinyTorch`.
- The `backend/cuda` folder contains the CUDA implementations of the operators.
- The `basic` folder contains basic data structures and functions.
- The `op` folder contains operators. Some of the operators have a forward and a backward function (e.g. `conv2d_forward`, `conv2d_backward`).
- The `pybind` folder contains bindings.
- The `tensor` folder contains the class `Tensor`, which is binded to class `TensorBase` in Python.
- The `utils` folder contains helper constants and functions. Note that `assert` won't be evaluated in release mode, it contains the definition of `ASSERT`.

The `unittest` folder contains unittests of the modules.

`basic.py` contains the class `Tensor` , the basic class `TensorOp`, and several subclasses of `TensorOp`. `Tensor` further encapsulates `TensorBase` to store its inputs and gradient. Operators encapsulated as subclasses of `TensorOp` can perform automatic differentiation.

`autodiff.py` contains the functions needed to implement automatic differentiation, basically the same as those in previous lab.

`module.py` contains the base class `Module` and several subclasses. All models should subclass the class `Module`, which is same as PyTorch's design. What is different is that the method `parameters()` returns an iterator that recursively traverses all `Tensor`s in the class.

`optimizer.py` contains the base class `Optimizer` and the class `SGD`, which can perform gradient descent with momentum.

`utils.py` contains some helpful functions to initialize Tensor.

## Usage

You can refer to code under the `cnn_mnist` or `mlp_mnist` folder for some details.
