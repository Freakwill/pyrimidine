#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BaseIndividual
from .chromosome import BinaryChromosome, BaseChromosome, FloatChromosome
from .meta import MetaTuple, MetaList


class MultiIndividual(BaseIndividual, metaclass=MetaTuple):
    pass

class MonoIndividual(BaseIndividual, metaclass=MetaList):
    """base class of individual with one choromosome

    You should implement the methods, cross, mute
    """
    n_chromosomes = 1

    @classmethod
    def random(cls, *args, **kwargs):
        return cls([cls.element_class.random(*args, **kwargs)])

    @property
    def chromosome(self):
        return self.chromosomes[0]

    def __iter__(self):
        return iter(self.chromosome)


class MonoBinaryIndividual(MonoIndividual):
    """simple binary individual
    encoded as a sequence such as 010011

    Equiv. to `MonoBinaryIndividual = MonoIndividual[BinaryChromosome]`
    """

    element_class = BinaryChromosome

class BinaryIndividual(MultiIndividual):
    """non-simple binary individual
    """

    element_class = BinaryChromosome,

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

class MixIndividual(BaseIndividual, metaclass=MetaTuple):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = BaseChromosome, BaseChromosome

    @classmethod
    def random(cls, sizes=(8, 8), n_chromosomes=None, size=None, *args, **kwargs):
        if sizes is None:
            sizes = (size,) * n_chromosomes
        else:
            if len(sizes) != len(cls.element_class):
                print(Warning('the length of sizes is not equal to the number of elements'))
        return cls([C.random(size=l, *args, **kwargs) for C, l in zip(cls.element_class, sizes)])


class AgeIndividual(BaseIndividual):
    age = 0
    life_span = 100  # life span

