#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from . import BaseGene


class NaturalGene(np.int_, BaseGene):
    lb, ub = 0, 10

    @classmethod
    def _random(cls):
        return cls(random.randint(cls.ub))

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(np.random.randint(cls.ub, *args, **kwargs))

class BinaryGene(np.int_, BaseGene):
    values = (0, 1)

    @classmethod
    def _random(cls):
        return cls(random.randint(2))

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(np.random.randint(2, *args, **kwargs))


class FloatGene(np.float_, BaseGene):
    lb, ub = 0, 1

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(np.random.uniform(cls.lb, cls.ub, *args, **kwargs))


class UnitFloatGene(FloatGene):
    lb, ub = 0, 1


class PeriodicGene(FloatGene):

    @property
    def period(self):
        if self.__period is None:
            self.__period = self.ub - self.lb
        return self.__period

class CircleGene(FloatGene):
    lb, ub = -np.pi, np.pi
    __period = 2 * np.pi

    @property
    def period(self):
        return self.__period

    
class SemiCircleGene(CircleGene):
    lb, ub = 0, np.pi
    __period = np.pi
