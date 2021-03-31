#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats

from .base import BaseChromosome, BaseGene
from .gene import *
from .utils import *


class ArrayChromosome(np.ndarray, BaseChromosome):
    element_class = BaseGene

    def __new__(cls, array, gene=None):
        if gene is None:
            gene = cls.element_class

        obj = super(ArrayChromosome, cls).__new__(cls, shape=array.shape, dtype=gene)
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.gene = getattr(obj, 'gene', None)

    def __len__(self):
        return self.shape[0]

    @property
    def n_genes(self):
        return len(self)
    
    @property
    def gene(self):
        return self.element_class

    @gene.setter
    def gene(self, ec):
        self.element_class = ec

    @classmethod
    def random(cls, *args, **kwargs):
        if 'size' not in kwargs:
            if cls.default_size:
                kwargs['size'] = cls.default_size
            else:
                raise UnknownSizeError(cls)
        return cls(array=cls.element_class.random(*args, **kwargs))

    def __str__(self):
        return f'{"|".join(str(gene) for gene in self)}'

    def cross(self, other):
        # note that when len(self) == 2 => k==1
        k = randint(1, len(self)-1)
        return self.__class__(array=np.concatenate((self[:k], other[k:]), axis=0), gene=self.gene)

    def merge(self, *other):
        return self

    def clone(self, *args, **kwargs):
        return self.__class__(array=self.copy(), gene=self.gene)

    # def mutate(self, indep_prob=0.1):
    #     for i in range(len(self)):
    #         if random() < indep_prob:
    #             self[i] = self.gene.random()


class VectorChromosome(ArrayChromosome):
    element_class = BaseGene

    # def __new__(cls, array, gene=None):
    #     if gene is None:
    #         gene = cls.element_class
    #     obj = super(VectorChromosome, cls).__new__(cls, shape=(len(array),), dtype=gene)
    #     obj = np.asarray(array).view(cls)
    #     obj.__gene = gene
    #     return obj

    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random() < indep_prob:
                self[i] = self.gene.random()


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
        return self.__class__(array=np.hstack((A, B)), gene=self.gene)


class BinaryChromosome(VectorChromosome):
    element_class = BinaryGene

    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random()< indep_prob:
                self[i] ^= 1

    def __str__(self):
        return f'{"".join(str(gene) for gene in self)}'

    def dual(self):
        return BinaryChromosome(1 ^ self)


class NaturalChromosome(VectorChromosome):
    element_class = NaturalGene

    def mutate(self, indep_prob=0.1):
        for i in range(len(self)):
            if random()< indep_prob:
                self[i] = NaturalGene.random()

    def __str__(self):
        return "".join(str(gene) for gene in self)

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
        return self.__class__(array=np.hstack((self[:k], [g for g in other if g not in self[:k]])), gene=self.gene)

    def __str__(self):
        if len(self)>10:
            return "|".join(str(gene) for gene in self)
        return "".join(str(gene) for gene in self)

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

    def random_neighbour(self, mu=0, simga=None):
        # select a neighour randomly
        sigma = sigma or self.sigma
        cpy = self.clone()
        n = scipy.stats.norm(mu, sigma)
        cpy += n.rvs(len(self))
        return cpy

class FloatMatrixChromosome(MatrixChromosome, FloatChromosome):
    pass

class PositiveChromosome(FloatChromosome):
    def max0(self):
        self[:] = max0(self)


class UnitFloatChromosome(PositiveChromosome):
    element_class = UnitFloatGene

    def dual(self):
        return UnitFloatChromosome(1 - self)

    def tobinary(self):
        return self >= 0.5

    def mutate(self, *args, **kwargs):
        super(UnitFloatChromosome, self).mutate(*args, **kwargs)
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
        return cls(np.random.dirichlet(np.ones(size)), gene=cls.element_class)


    def check(self):
        raise np.sum(self) == 1

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
        array=np.hstack((self[:k], other[k:]))
        array /= array.sum()
        return self.__class__(array=array, gene=self.gene)

    def random_neighbour(self):
        # select a neighour randomly
        cpy = self.clone()
        i, j = randint2(0, len(cpy)-1)
        p = cpy[i] + cpy[j]
        r = np.random.uniform(0, p)
        cpy[i] = r
        cpy[j] = p-r
        return cpy


    def mutate(self, *args, **kwargs):
        super(ProbabilityChromosome, self).mutate(*args, **kwargs)
        self.normalize()

    def normalize(self):
        self.max0()
        self /= self.sum()


class CircleChromosome(FloatChromosome):
    """Used in Quantum-Chromosome
    
    Extends:
        FloatChromosome
    """
    element_class = CircleGene

    def mutate(self, *args, **kwargs):
        super(CircleChromosome, self).mutate(*args, **kwargs)
        self.normalize()

    def normalize(self):
        self %= self.element_class.period


class QuantumChromosome(CircleChromosome):
    measure_result = None
    def decode(self):
        rs = np.random.random(size=(len(self),))
        self.measure_result = self.decode()
        return np.cos(self) ** 2 < rs
