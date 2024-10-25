#!/usr/bin/env python3

"""
Individual classes

An individual represents a solution of a optimization problem.
It is defined as a container of chromosomes.
"""

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


from .deco import fitness_cache


@fitness_cache
class PhantomIndividual(BaseIndividual):
    # Anthor way to implement memory system for individual

    phantom = None

    def init(self):
        self.phantom = self.copy()

    def backup(self):
        if self.fitness < self.phantom.fitness:
            self.chromosomes = self.phantom.chromosomes
            self.set_cache(fitness=self.phantom.fitness)


# Following are functions to create individuals

def binaryIndividual(size=8):
    """simple binary individual
    encoded as a sequence such as 01001101

    Equiv. to `makeIndividual(size=size)`
    """

    return MonoIndividual[BinaryChromosome.set(default_size=size)]


def makeIndividual(element_class=BinaryChromosome, n_chromosomes=None, size=8, cls=None):
    """helper to make an individual
    
    Example:
        # make an indiviudal with 2 binary chromosomes, each chromosome has 8 genes,
        makeIndividual(element_class=BinaryChromosome, n_chromosomes=2, size=8)
        # make mixed indiviudal with 2 type of chromosomes, one has 8 genes and the other has 2,
        makeIndividual(element_class=(BinaryChromosome, FloatChromosome), size=(8,2))
    
    Args:
        element_class (BaseChromosome, tuple, optional): class of chromosomes
        n_chromosomes (int, optional): the number of chromosomes
        size (int, tuple, optional): the sizes of chromosomes
        cls: the class of the return individual
    
    Returns:
        cls or other individual classes
    """

    if n_chromosomes is None:
        if isinstance(size, tuple):
            n_chromosomes = len(size)
        else:
            n_chromosomes = 1

    if n_chromosomes == 1:
        assert isinstance(size, int)
        cls = cls or MonoIndividual
        return cls[element_class // size]
    else:
        if isinstance(size, tuple):
            if len(size) == n_chromosomes:
                if isinstance(element_class, tuple):
                    cls = cls or MixedIndividual
                    return cls[tuple(e_c // s for e_c, s in zip(element_class, size))]
                else:
                    cls = cls or MixedIndividual
                    return cls[tuple(element_class // s for s in size)]
            else: 
                raise ValueError('the length of `size` must be `n_chromosomes`.')
        elif isinstance(size, int):
            cls = cls or PolyIndividual
            return cls[element_class // size].set(default_size=n_chromosomes)
        else:
            raise ValueError('the length of `size` must be a number or a tuple of numbers.')


def makeBinaryIndividual(size=8, cls=None):
    """To make a binary individual

    Examples:
        makeBinaryIndividual(size=8) # an individual containting one binary choromosome with 8 genes
    """

    cls = cls or MixedIndividual
    if isinstance(size, tuple):
        return cls[tuple(BinaryChromosome // s for s in size)]
    else:
        return cls[BinaryChromosome // size]

