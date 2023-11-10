#!/usr/bin/env python3

import numpy as np
import scipy.stats

from .base import BaseChromosome, BaseGene
from .gene import *
from .utils import *


def _asarray(out):
    return np.asarray(out) if isinstance(out, np.ndarray) else out


class ArrayChromosome(BaseChromosome, np.ndarray):

    """Summary
    
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

    def __len__(self):
        return self.size

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
        return cls(array=cls.element_class.random(*args, **kwargs))

    def __str__(self):
        return f'{"|".join(map(str, self))}'

    def cross(self, other):
        # note that when len(self) == 2  ==>  k==1
        k = randint(1, len(self)-1)
        return self.__class__(np.concatenate((self[:k], other[k:]), axis=0), element_class=self.element_class)

    def merge(self, *other):
        return self

    def clone(self, *args, **kwargs):
        return self.__class__(self.copy(), element_class=self.element_class)

    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] = self.element_class.random()


class VectorChromosome(ArrayChromosome):
    element_class = BaseGene


class MatrixChromosome(ArrayChromosome):
    
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
        return self.__class__(np.hstack((A, B)), element_class=self.element_class)


class BinaryChromosome(VectorChromosome):
    element_class = BinaryGene

    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random()< indep_prob:
                self[i] ^= 1

    def __str__(self):
        return f'{"".join(map(str, self))}'

    def dual(self):
        return BinaryChromosome(1 ^ self)


class NaturalChromosome(VectorChromosome):
    element_class = NaturalGene

    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random()< indep_prob:
                self[i] = NaturalGene.random()

    def __str__(self):
        return "".join(map(str, self))

    def dual(self):
        return NaturalChromosome(self.element_class.ub - self)


class PermutationChromosome(NaturalChromosome):
    element_class = NaturalGene
    default_size = 10

    @classmethod
    def random(cls, size=None):
        size = size or cls.default_size
        return cls(np.random.permutation(cls.default_size))

    # def mutate(self):
    #     i = randint(0, self.default_size-1)
    #     if i == self.default_size-1:
    #         j = 0
    #     else:
    #         j = i+1
    #     t = self[i]; self[i] = self[j]; self[j] = t

    def mutate(self):
        i, j = randint2(0, self.default_size-1)
        t = self[i]; self[i] = self[j]; self[j] = t

    def cross(self, other):
        k = randint(1, len(self)-2)
        return self.__class__(np.hstack((self[:k], [g for g in other if g not in self[:k]])), element_class=self.element_class)

    def __str__(self):
        if len(self)>10:
            return "|".join(map(str, self))
        return "".join(map(str, self))

    def dual(self):
        return NaturalChromosome(self.element_class.ub - self)


class FloatChromosome(VectorChromosome):
    element_class = FloatGene
    sigma = 0.05

    def mutate(self, indep_prob=0.1, mu=0, sigma=None):
        sigma = sigma or self.sigma
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] += gauss(mu, sigma)
        return self

    def random_neighbour(self, mu=0, sigma=None):
        # select a neighour randomly
        sigma = sigma or self.sigma
        cpy = self.copy()
        n = scipy.stats.norm(mu, sigma)
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
        FloatChromosome
    """

    element_class = UnitFloatGene

    @classmethod
    def random(cls, size=None):
        if size is None:
            if cls.default_size:
                size = cls.default_size
            else:
                raise UnknownSizeError(cls)
        return cls(np.random.dirichlet(np.ones(size)), element_class=cls.element_class)


    # def check(self):
    #     assert np.sum(self) == 1, 'the sum of the chromosome must be 1!'

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
        return self.__class__(array, element_class=self.element_class)

    def random_neighbour(self):
        # select a neighour randomly
        cpy = self.copy()
        i, j = randint2(0, len(cpy)-1)
        p = cpy[i] + cpy[j]
        r = np.random.uniform(0, p)
        cpy[i] = r
        cpy[j] = p-r
        return cpy


    def mutate(self, *args, **kwargs):
        super().mutate(*args, **kwargs)
        self.normalize()

    def normalize(self):
        super().normalize()
        a = self + 0.001
        self[:] = a / np.sum(a)


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
        self %= self.element_class.period


class QuantumChromosome(CircleChromosome):
    measure_result = None
    def decode(self):
        rs = np.random.random(size=(len(self),))
        self.measure_result = self.decode()
        return np.cos(self) ** 2 < rs
