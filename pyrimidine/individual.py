#!/usr/bin/env python3

"""
Individual classes
"""

from .base import BaseIndividual
from .chromosome import BinaryChromosome, BaseChromosome, FloatChromosome
from .meta import MetaTuple, MetaList
from .utils import randint
import copy


class MultiIndividual(BaseIndividual, metaclass=MetaList):
    pass


PolyIndividual = MultiIndividual


class MonoIndividual(BaseIndividual, metaclass=MetaList):
    """Base class of individual with one choromosome

    You should implement the methods, cross, mute
    """

    n_chromosomes = 1

    @classmethod
    def random(cls, *args, **kwargs):
        return cls([cls.element_class.random(*args, **kwargs)])

    @property
    def chromosome(self):
        return self.chromosomes[0]

    @chromosome.setter
    def chromosome(self, v):
        self.chromosomes[0] = v

    def decode(self):
        return self.chromosome.decode()

    @classmethod
    def set_size(cls, sz):
        raise DeprecationWarning("Never set the size of this class!")

    @property
    def individuals(self):
        return self.__elements

    @individuals.setter
    def individuals(self, x):
        if len(x)>=2:
            raise ValueError('A monoIndividual has only one chromosome! But you give more than one.')
        super().individuals = x


def classicalIndividual(size=8):
    """simple binary individual
    encoded as a sequence such as 01001101
    """
    return MonoIndividual[BinaryChromosome.set(default_size=size)]


def makeIndividual(element_class=BinaryChromosome, n_chromosomes=1, size=8):
    """helper to make an individual
    """
    if n_chromosomes == 1:
        return MonoIndividual[FloatChromosome.set(default_size=size)]
    else:
        if isinstance(size, tuple):
            if len(size) == n_chromosomes:
                return PolyIndividual[tuple(BinaryChromosome.set(default_size=s) for s in size)]
            elif len(size) == 1:
                return PolyIndividual[tuple(BinaryChromosome.set(default_size=size[0]) for _ in n_chromosomes)]
            else:
                raise ValueError('The length of size must be 1 or `n_chromosomes`.')
        else:
            return PolyIndividual[tuple(BinaryChromosome.set(default_size=size) for _ in np.range(n_chromosomes))]


class MixedIndividual(BaseIndividual, metaclass=MetaTuple):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = BaseChromosome, BaseChromosome

    @classmethod
    def random(cls, sizes=None, n_chromosomes=None, size=None, *args, **kwargs):
        if sizes is None:
            if isinstance(size, int):
                sizes = (size,) * n_chromosomes
            elif size is None:
                return cls([C.random(*args, **kwargs) for C in cls.element_class])
            else:
                raise TypeError('Argument `size` should be an integer or None(by default)!')
        else:
            if len(sizes) != len(cls.element_class):
                print(Warning('the length of sizes is not equal to the number of elements'))
        return cls([C.random(size=l, *args, **kwargs) for C, l in zip(cls.element_class, sizes)])


class AgeIndividual(BaseIndividual):
    age = 0
    life_span = 100


class GenderIndividual(MixedIndividual):

    @property
    def gender(self):
        raise NotImplementedError


class MemoryIndividual(BaseIndividual):
    _memory = {"fitness": None}

    @property
    def memory(self):
        return self._memory

    def init(self, fitness=True, type_=None):
        self.backup(check=False)

    def backup(self, check=False):
        if not check or self.memory['fitness'] is None or self._fitness() > self.memory['fitness']:
            def _map(k):
                if k == 'fitness':
                    return self._fitness()
                elif hasattr(getattr(self, k), 'copy'):
                    return getattr(self, k).copy()
                elif hasattr(getattr(self, k), 'clone'):
                    return getattr(self, k).clone()
                else:
                    return getattr(self, k)
            self._memory = {k: _map(k) for k in self.memory.keys()}

    @property
    def fitness(self):
        if 'fitness' in self.memory and self.memory['fitness'] is not None:
            return self.memory['fitness']
        else:
            return super().fitness

    def clone(self, *args, **kwargs):
        cpy = super().clone(*args, **kwargs)
        cpy._memory = self.memory
        return cpy


class PhantomIndividual(BaseIndividual):
    phantom = None

    def init(self, fitness=True, type_=None):
        self.phantom = self.clone(fitness=fitness, type_=type_)

    def backup(self):
        if self.fitness < self.phantom.fitness:
            self.chromosomes = self.phantom.chromosomes
            self.fitness = self.phantom.fitness
