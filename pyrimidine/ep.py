#!/usr/bin/env python3

"""Evolution Programming

Invented by L. Fogel[1966] for designing FSM initially.

General form of the algorithm:

    1. initialize a population with N individuals
    2. loop:
        calculate f(x) for each x in population
        mutate x for each x
        get new population (mixed with the original population)
        select best N individuals from the 2N mixed population

    where the mutation defined as,
        x' <- x + r*sqrt(v)
        v' <- v + c*r*sqrt(v) (make sure that v>epsilon)

Remark:
    No cross operation in EP
"""

from random import choice
from toolz.itertoolz import groupby
from operator import attrgetter

import numpy as np

from .base import BasePopulation, BaseChromosome
from .chromosome import FloatChromosome
from .individual import MixedIndividual

from .deco import side_effect, fitness_cache


class BaseEPIndividual(MixedIndividual):
    """Base class of EP Individual Class

    A single solution in EP
    """

    params = {'c':0.1, 'epsilon':0.0001}

    element_class = BaseChromosome, FloatChromosome

    def decode(self):
        return self.chromosomes[0]

    @property
    def variance(self):
        return self.chromosomes[1]

    @variance.setter
    def variance(self, v):
        self.chromosomes[1] = v
    
    @side_effect
    def mutate(self):
        rx = np.random.randn(*self.chromosomes[0].shape)
        self.chromosomes[0] += rx * np.sqrt(self.variance)

        rv = np.random.randn(*self.variance.shape)
        self.variance += self.c * rv * np.sqrt(self.variance)
        self.variance = np.maximum(self.variance, self.epsilon)


class EvolutionProgramming(BasePopulation):
    """Evolution Programming
    
    Extends:
        PopulationMixin
    """
    
    element_class = BaseEPIndividual

    def select(self):
        self.sort()
        d = groupby(attrgetter('fitness'), self)
        inds = []
        ks = np.sort(list(d.keys()))[::-1]
        flag = True
        while flag:
            for k in ks:
                if d[k]:
                    a = choice(d[k])
                    inds.append(a)
                    d[k].remove(a)
                    if len(inds) <= self.default_size:
                        flag = False
                        break
        self.individuals = inds

    def transition(self, *args, **kwargs):
        cpy = self.clone()
        self.mutate()
        self.merge(cpy)
        self.select()
    
