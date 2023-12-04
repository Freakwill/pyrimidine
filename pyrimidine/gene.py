#!/usr/bin/env python3

"""
Gene classes
"""

import numpy as np
from . import BaseGene


class NaturalGene(np.int_, BaseGene):
    lb, ub = 0, 10

    @classmethod
    def random(cls, *args, **kwargs):
        return np.random.randint(cls.ub, dtype=cls, *args, **kwargs)


class DigitGene(NaturalGene):
    pass


class IntegerGene(np.int_, BaseGene):
    lb, ub = -10, 10

    @classmethod
    def random(cls, *args, **kwargs):
        return np.random.randint(cls.ub-cls.lb, dtype=cls, *args, **kwargs) + cls.lb


class BinaryGene(np.int_, BaseGene):
    values = (0, 1)

    @classmethod
    def random(cls, *args, **kwargs):
        return np.random.randint(2, dtype=cls, *args, **kwargs)


class FloatGene(np.float_, BaseGene):
    lb, ub = 0, 1

    @classmethod
    def random(cls, *args, **kwargs):
        if 'size' in kwargs:
            return np.random.uniform(cls.lb, cls.ub, *args, **kwargs).astype(cls, copy=False)
        else:
            return np.random.uniform(cls.lb, cls.ub, *args, **kwargs)


class UnitFloatGene(FloatGene):
    lb, ub = 0, 1


class PeriodicGene(FloatGene):

    @property
    def period(self):
        return ub - lb


class CircleGene(PeriodicGene):
    lb, ub = 0, 2*np.pi
    period = 2*np.pi

    
class SemiCircleGene(CircleGene):
    lb, ub = 0, np.pi
