#!/usr/bin/env python3

# from .approximation import *
# from .cluster import *
# from .fitting import *
# from .linear_model import *
# from .matrix import *
# from .neural_network import *
# from .optimization import *
# from .special import *


class BaseProblem:
    """Base Class of Problems

    The class is just a function (callable object).
    Please implement `__call__` method in each subclass
    make it behaving as function

    In fact, it is never used, just a template.
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError