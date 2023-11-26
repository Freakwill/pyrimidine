#!/usr/bin/env python3

"""
Individual classes
"""

import copy

from .base import BaseIndividual
from .chromosome import BinaryChromosome, BaseChromosome, FloatChromosome
from .meta import MetaTuple, MetaList, MetaSingle
from .utils import randint


class MultiIndividual(BaseIndividual, metaclass=MetaList):
    pass


PolyIndividual = MultiIndividual


class MonoIndividual(BaseIndividual, metaclass=MetaSingle):
    """Base class of individual with only one chromosome;
    It is equavalent to a chromosome.

    You should implement the genetic operations: cross, mutate.
    """

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


class MixedIndividual(BaseIndividual, metaclass=MetaTuple):
    """base class of individual

    You should implement the enetic operations: cross, mutate
    """

    element_class = BaseChromosome, BaseChromosome

    @property
    def default_size(self):
        return len(self.element_class)

    @classmethod
    def random(cls, size=None, n_chromosomes=None, *args, **kwargs):
        if size is None:
            return cls([C.random(*args, **kwargs) for C in cls.element_class])
        elif isinstance(size, int):
            size = (size,) * n_chromosomes
        else: #if isinstance(size, tuple):
            if len(size) != len(cls.element_class):
                raise ValueError('the length of `size` is not equal to the number of elements (`n_chromosomes`)')
        return cls([C.random(size=l, *args, **kwargs) for C, l in zip(cls.element_class, size)])


class AgeIndividual(BaseIndividual):

    params = {
    "age": 0,
    "life_span": 100
    }


class GenderIndividual(MixedIndividual):

    @property
    def gender(self):
        raise NotImplementedError


class MemoryIndividual(BaseIndividual):
    # Individual with memory, used in PSO

    _memory = {"fitness": None}

    @property
    def memory(self):
        return self._memory

    def init(self, fitness=True, type_=None):
        self.backup(check=False)

    def backup(self, check=False):
        """Backup the fitness and other information
        
        Args:
            check (bool, optional): check whether the fitness increases.
        """
        
        if not check or (self.memory['fitness'] is None or self._fitness() > self.memory['fitness']):
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
    # Another implementation of the individual class with memory

    phantom = None

    def init(self, fitness=True, type_=None):
        self.phantom = self.clone(fitness=fitness, type_=type_)

    def backup(self):
        if self.fitness < self.phantom.fitness:
            self.chromosomes = self.phantom.chromosomes
            self.__fitness = self.phantom.fitness


# Following are functions to create individuals

def binaryIndividual(size=8):
    """simple binary individual
    encoded as a sequence such as 01001101

    Equiv. to `makeIndividual(size=size)`
    """

    return MonoIndividual[BinaryChromosome.set(default_size=size)]


makeBinaryIndividual = binaryIndividual


def makeIndividual(cls=None, element_class=BinaryChromosome, n_chromosomes=1, size=8):
    """helper to make an individual
    
    Example:
        # make an indiviudal with 2 binary chromosomes, each chromosome has 8 genes,
        makeIndividual(element_class=BinaryChromosome, n_chromosomes=2, size=8)
        # make mixed indiviudal with 2 type of chromosomes, one has 8 genes and the other has 2,
        makeIndividual(element_class=(BinaryChromosome, FloatChromosome), size=(8,2))
    
    Args:
        element_class (BaseChromosome, tuple, optional): class of chromosomes
        n_chromosomes (int, optional): number of chromosomes
        size (int, tuple, optional): the sizes of chromosomes
    
    Returns:
        BaseIndividual
    """

    if n_chromosomes == 1:
        assert isinstance(size, int)
        cls = cls or MonoIndividual
        return cls[element_class.set(default_size=size)]
    else:
        if isinstance(size, tuple):
            if len(size) == n_chromosomes:
                if isinstance(element_class, tuple):
                    cls = cls or MixedIndividual
                    return cls[tuple(e_c.set(default_size=s) for e_c, s in zip(element_class, size))]
                else:
                    cls = cls or MixedIndividual
                    return cls[tuple(element_class.set(default_size=s) for s in size)]
            elif len(size) == 1:
                cls = cls or PolyIndividual
                return PolyIndividual[tuple(element_class.set(default_size=size[0]) for _ in n_chromosomes)]
            else:
                raise ValueError('the length of `size` must be 1 or `n_chromosomes`.')
        elif isinstance(size, int):
            cls = cls or PolyIndividual
            return cls[tuple(element_class.set(default_size=size) for _ in np.range(n_chromosomes))]
        else:
            raise ValueError('the length of `size` must be a number or a tuple of numbers.')
