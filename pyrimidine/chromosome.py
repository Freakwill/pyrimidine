#!/usr/bin/env python3

"""
Chromosome classes, subclass of BaseChromosome

A chromosome is an array of genes, or it can be customized by the user.
It could be a part of an individual or encodes a solution directly.
"""

from random import choice, randint, gauss, random

import numpy as np
from scipy.stats import norm
from scipy.special import softmax

from .base import BaseChromosome, BaseGene
from .gene import *
from .utils import *
from .deco import side_effect


def _asarray(out):
    return np.asarray(out) if isinstance(out, np.ndarray) else out


class NumpyArrayChromosome(BaseChromosome, np.ndarray):
    """Chromosome implemented by `np.array`
    
    Attributes:
        element_class (TYPE): the type of gene
    """

    element_class = BaseGene

    def __new__(cls, array=None, element_class=None):
        if element_class is None:
            element_class = cls.element_class
        if array is None:
            array = []

        return np.asarray(array, dtype=element_class).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if isinstance(obj, (tuple, list)):
            obj = self.__class__(obj)
        if isinstance(obj, BaseChromosome):
            self.element_class = getattr(obj, 'element_class', None)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):

        inputs = map(_asarray, inputs)
        if out is not None:
            out = tuple(map(_asarray, out))

        results = super().__array_ufunc__(ufunc, method, *inputs, out=out, **kwargs)

        if results is NotImplemented:
            return NotImplemented
        elif ufunc.nout == 1:
            if out is None:
                return self.__class__(results) if isinstance(results, np.ndarray) else results
            else:
                out = out[0]
                return self.__class__(out) if isinstance(out, np.ndarray) else out
        else:
            if out is None:
                return tuple(map(lambda o: self.__class__(o) if isinstance(o, np.ndarray) else o, results))
            else:
                return tuple(map(lambda o: self.__class__(o) if isinstance(o, np.ndarray) else o, out))

    @property
    def elements(self):
        return np.asarray(self)

    @elements.setter
    def elements(self, x):
        self.__elements = np.asarray(x)

    @classmethod
    def random(cls, *args, **kwargs):
        if 'size' not in kwargs:
            if cls.default_size:
                kwargs['size'] = cls.default_size
            else:
                raise UnknownSizeError(cls)
        return cls(cls.element_class.random(*args, **kwargs))

    def cross(self, other):
        """Crossover operation for a single chromosome

        Note that when len(self) == 2  ==>  k==1
        
        Args:
            other (BaseChromosome): another chromosome
        
        Returns:
            BaseChromosome: new chromosome, as the child of the two chromosomes
        """

        k = randint(1, len(self)-1)
        return self.__class__(np.concatenate((self[:k], other[k:]), axis=0))

    def cross2(self, other):
        # return 2 children after crossover
        k = randint(1, len(self)-1)
        return (self.__class__(np.concatenate((self[:k], other[k:]), axis=0)),
                self.__class__(np.concatenate((other[:k], self[k:]), axis=0)))

    def copy(self, type_=None, *args, **kwargs):
        type_ = type_ or self.__class__
        return type_(np.copy(self))

    def clone(self):
        """Alias of `copy`, but regarded as a genetic operation
        
        Returns:
            BaseChromosome
        """

        return self.__class__(np.copy(self))

    @side_effect
    def mutate(self, indep_prob=0.1):
        # mutation
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] = self.element_class.random()


class VectorChromosome(NumpyArrayChromosome):
    pass


class MatrixChromosome(NumpyArrayChromosome):
    
    @side_effect
    def mutate(self, indep_prob=0.1):
        r, c = self.shape
        for i in range(r):
            for j in range(c):
                if random() < indep_prob:
                    self[i, j] += gauss(0, 0.1)

    def cross(self, other):
        r, c = self.shape
        k, l = randint(1, r-1), randint(1, c-1)
        A = np.vstack((self[:k, :l], other[k:, :l]))
        B = np.vstack((other[:k, l:], self[k:, l:]))
        return self.__class__(np.hstack((A, B)))


class NaturalChromosome(VectorChromosome):

    element_class = NaturalGene

    @side_effect
    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random()< indep_prob:
                self[i] = NaturalGene.random()

    def dual(self):
        return self.__class__(self.element_class.ub - self)


class DigitChromosome(NaturalChromosome):

    element_class = DigitGene

    def __str__(self):
        return "".join(map(str, self))


class BinaryChromosome(NaturalChromosome):

    element_class = BinaryGene

    def __str__(self):
        return "".join(map(str, self))

    @classmethod
    def zero(cls):
        return cls(np.zeros(cls.default_size))

    @classmethod
    def one(cls):
        return cls(np.ones(cls.default_size))

    @side_effect
    def mutate(self, indep_prob=0.5):
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] ^= 1

    def dual(self):
        return self.__class__(1 ^ self)


class PermutationChromosome(NaturalChromosome):
    # A chromosome representing a permutation

    element_class = NaturalGene
    default_size = 10

    @classmethod
    def identity(cls):
        return cls(np.arange(cls.default_size))

    @classmethod
    def random(cls, size=None):
        size = size or cls.default_size
        return cls(np.random.permutation(cls.default_size))

    # def __sub__(self, other):
    #     return rotations(self, other)

    def move_toward(self, other):
        r = choice(rotations(self, other))
        rotate(self, r)

    @side_effect
    def mutate(self):
        i, j = randint2(0, self.default_size-1)
        self[[i,j]] = self[[j,i]]

    def cross(self, other):
        k = randint(1, len(self)-2)
        return self.__class__(np.hstack((self[:k], [g for g in other if g not in self[:k]])))

    def __str__(self):
        if len(self)>10:
            return ",".join(map(str, self))
        return "".join(map(str, self))

    def dual(self):
        return self[::-1]


class FloatChromosome(NumpyArrayChromosome):

    element_class = FloatGene
    sigma = 0.1

    def __str__(self):
        return "|".join(format(c, '.4') for c in self)

    @side_effect
    def mutate(self, indep_prob=0.1, mu=0, sigma=None):
        sigma = sigma or self.sigma
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] += gauss(mu, sigma)

    def random_neighbour(self, mu=0, sigma=None):
        # select a neighour randomly
        sigma = sigma or self.sigma
        cpy = self.copy()
        n = norm(mu, sigma)
        cpy += n.rvs(len(self))
        return cpy


class FloatMatrixChromosome(MatrixChromosome, FloatChromosome):
    pass


class PositiveChromosome(FloatChromosome):

    def normalize(self):
        self[:] = max0(self)


class UnitFloatChromosome(PositiveChromosome):

    element_class = UnitFloatGene

    def dual(self):
        return UnitFloatChromosome(1 - self)

    def tobinary(self):
        return self >= 0.5

    def mutate(self, *args, **kwargs):
        super().mutate(*args, **kwargs)
        self.normalize()

    def normalize(self):
        self[:] = hl(self)


class ProbabilityChromosome(PositiveChromosome):
    """ProbabilityChromosome

    The genes represent a distribution, such as [0.1,0.2,0.3,0.4].
    
    Extends:
        PositiveChromosome
    """

    element_class = UnitFloatGene

    @classmethod
    def random(cls, size=None):
        if size is None:
            if cls.default_size:
                size = cls.default_size
            else:
                raise UnknownSizeError(cls)
        return cls(np.random.dirichlet(np.ones(size)))

    def check(self):
        assert np.sum(self) == 1, 'the sum of the chromosome must be 1!'

    # def mutate(self, indep_prob=0.1):
    #     """Mutation of ProbabilityChromosome
    #     if mutation happend on two genes i and j, then select a number r randomly
    #     i <- r, j <= p - r, where p is the sum of the original probs of i and j.
        
    #     Keyword Arguments:
    #         indep_prob {number} -- independent prob of mutation for any gene (default: {0.1})
        
    #     Returns:
    #         ProbabilityChromosome -- new obj after mutation
    #     """
    #     for i in range(len(self)-1):
    #         if random() < indep_prob:
    #             j = randint(i+1, len(self)-1)
    #             p = self[i] + self[j]
    #             r = np.random.uniform(0, p)
    #             self[i] = r
    #             self[j] = p-r
    #     return self

    def cross(self, other):
        k = randint(1, len(self)-2)
        array = np.hstack((self[:k], other[k:])) + 0.001
        array /= array.sum()
        return self.__class__(array)

    def random_neighbour(self):
        # select a neighour randomly
        cpy = self.copy()
        i, j = randint2(0, len(cpy)-1)
        p = cpy[i] + cpy[j]
        r = random() * p
        cpy[[i,j]] = [r, p - r]
        return cpy

    def mutate(self, *args, **kwargs):
        super().mutate(*args, **kwargs)
        self.normalize()

    def normalize(self):
        self[:] = softmax(self)


class StochasticMatrixChromosome(MatrixChromosome, PositiveChromosome):

    def normalize(self):
        self[:] = softmax(self, axis=1)

class CircleChromosome(FloatChromosome):
    """Used in Quantum-Chromosome
    
    Extends:
        FloatChromosome
    """

    element_class = CircleGene

    def mutate(self, *args, **kwargs):
        super().mutate(*args, **kwargs)
        self.normalize()

    def normalize(self):
        self[:] = self % self.element_class.period


class QuantumChromosome(CircleChromosome):
 
    _measure_result = None

    @property
    def measure_result(self):
        if self._measure_result is None:
            self.measure()
        return self._measure_result

    def measure(self):
        # measure a quantum chromosome to get a binary sequence
        rs = np.random.random(size=(len(self),))
        self._measure_result = np.asarray(np.cos(self) ** 2 > rs, dtype=np.int_)

    def decode(self):
        return self.measure_result


# Implement Chromosome class by `array.array`

import copy
import array

class ArrayChromosome(BaseChromosome, array.array):
    """Chromosome class implemented by `array.array`
    
    Attributes:
        element_class (str): the type of gene
    """

    element_class = 'd'

    def __new__(cls, array=None, element_class=None):
        if element_class is None:
            element_class = cls.element_class
        if array is None:
            array = []

        return array.array(element_class, array)

    def cross(self, other):
        # note that when len(self) == 2  ==>  k==1
        k = randint(1, len(self)-1)
        return self[:k] + other[k:]

    def copy(self, type_=None):
        return copy.copy(self)

    @side_effect
    def mutate(self, indep_prob=0.1):
        a = self.random()
        for k, ak in enumerate(a):
            if random() < indep_prob:
                self[k] = ak


class ListChromosome(BaseChromosome, list):
    """Chromosome class implemented by `list`
    
    Attributes:
        element_class (str): the type of gene
    """

    element_class = None

    def cross(self, other):
        # note that when len(self) == 2  ==>  k==1
        k = randint(1, len(self)-1)
        return self[:k] + other[k:]

    def copy(self, type_=None):
        return copy.deepcopy(self)

    @side_effect
    def mutate(self, indep_prob=0.1):
        a = self.random()
        for k, ak in enumerate(a):
            if random() < indep_prob:
                self[k] = ak
