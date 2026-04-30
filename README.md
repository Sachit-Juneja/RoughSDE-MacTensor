# RoughSDE-MacTensor

A state-of-the-art Generative Model for high-frequency financial data: a Neural Stochastic Differential Equation (Neural SDE) trained via Signature Kernels. 

The heavy mathematical lifting is implemented as a native C++ extension using the `MacTensor` library, which is exposed to Python via `pybind11` for Neural Network orchestration.

## Setup
```bash
mkdir build && cd build
cmake ..
make
```
