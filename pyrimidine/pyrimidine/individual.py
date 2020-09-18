#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import (BaseIndividual, 
    BinaryChromosome, BaseChromosome, FloatChromosome)

class MultiIndividual(BaseIndividual):
    pass

class SimpleIndividual(BaseIndividual):
    """base class of individual with one choromosome

    You should implement the methods, cross, mute
    """
    n_chromosomes = 1

    @classmethod
    def random(cls, *args, **kwargs):
        return cls(chromosomes=[cls.element_class.random(*args, **kwargs)])

    @property
    def chromosome(self):
        return self.chromosomes[0]

    def __iter__(self):
        return iter(self.chromosome)


class SimpleBinaryIndividual(SimpleIndividual):
    """simple binary individual
    encoded as a sequence such as 010011
    """

    element_class = BinaryChromosome

class BinaryIndividual(BaseIndividual):
    """simple binary individual
    encoded as a sequence such as 010011
    """

    element_class = BinaryChromosome

class FloatIndividual(BaseIndividual):
    """simple binary individual
    encoded as a sequence such as 010011
    """

    element_class = FloatChromosome

class SimpleFloatIndividual(SimpleIndividual):
    """simple binary individual
    encoded as a sequence such as 010011
    """

    element_class = FloatChromosome

class MixIndividual(BaseIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = BaseChromosome,

    @classmethod
    def random(cls, sizes=(8,), n_chromosomes=None, size=None, *args, **kwargs):
        if sizes is None:
            sizes = (size,) * n_chromosomes
        return cls(chromosomes=[C.random(size=l, *args, **kwargs) for C, l in zip(cls.element_class, sizes)])


class AgeIndividual(BaseIndividual):
    age = 0
    life_span = 100  # life span

