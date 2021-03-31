#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseIndividual
from .chromosome import BinaryChromosome, BaseChromosome, FloatChromosome
from .meta import MetaTuple, MetaList
from .utils import randint


class MultiIndividual(BaseIndividual, metaclass=MetaList):
    pass


class PolyIndividual(BaseIndividual, metaclass=MetaList):
    pass

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
        super(MonoIndividual, self).individuals = x


class MonoBinaryIndividual(MonoIndividual):
    """simple binary individual
    encoded as a sequence such as 010011

    Equiv. to `MonoBinaryIndividual = MonoIndividual[BinaryChromosome]`
    """

    element_class = BinaryChromosome

class BinaryIndividual(MultiIndividual):
    """non-simple binary individual
    """

    element_class = BinaryChromosome

class FloatIndividual(BaseIndividual):
    """simple binary individual
    encoded as a sequence such as 010011
    """

    element_class = FloatChromosome


class MonoFloatIndividual(MonoIndividual):
    """simple binary individual
    encoded as a sequence such as 010011
    """

    element_class = FloatChromosome

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
    life_span = 100  # life span


class GenderIndividual(MixedIndividual):

    @property
    def gender(self):
        raise NotImplementedError
