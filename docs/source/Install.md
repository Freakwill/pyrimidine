# Installation and Getting Started

**Pythons**: Python 3.9, 3.10, 3.11, 3.12, PyPy3 (even Python 3.13)

**Platforms**: Linux/Unix and Windows

**PyPI package name**: pyrimidine

`pyrimidine` stands as a versatile framework designed for GAs, offering exceptional extensibility for a wide array of evolutionary algorithms, including particle swarm optimization and difference evolution.

## Install pyrimidine

Run the following command to install `pyrimidine`:

`pip install [-U] pyrimidine`

Check the version:

`$ pyrimidine --version`

Note that you shold use `sudo pip3 install pyrimidine` to avoid the permission error.

## Requirements

`Pyrimidine` requires few packages. It only needs `numpy/scipy` for computing, and `pandas/matplotlib` for visualization. 

It is recommended to use `digit_converter` for decoding chromosomes to the real solutions, which is developed by the author for GA. However, users can implement the deoding method in your own method.

`ezstat` is also required for statistics which is also created by the author.

Both of these can be installed by including the "optional" key when installing:

`pip install pyrimidine[optional]`

## First test

```python
import pyrimidine
# or from pyrimidine import *
```

