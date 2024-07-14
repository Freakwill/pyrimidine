#!/usr/bin/env python

"""
Differential Evolution Algorithm

*Ref*
Price, Kenneth V., Rainer M. Storn, and Jouni A. Lampinen. The differential evolution algorithm. Differential evolution: a practical approach to global optimization (2005): 37-134.
"""


import numpy as np

from .base import BaseIndividual
from .mixin import PopulationMixin
from .meta import MetaContainer
from .individual import MonoIndividual

from .utils import *


class DifferentialEvolution(PopulationMixin, metaclass=MetaContainer):
    # Differential Evolution Algo.
    
    element_class = BaseIndividual
    default_size = 4

    params ={
        "factor" : 0.05,
        "cross_prob": 0.75,
    }

    def init(self):
        self.ndims = tuple(map(len, self[0]))

    def transition(self, *args, **kwargs):
        self.move()
        for k, (test_individual, individual) in enumerate(zip(self.test, self)):
            if test_individual.fitness > individual.fitness:
                self[k] = test_individual

    def move(self):
        self.test = self.copy()
        for t in self.test:
            x0, x1, x2 = self.get_samples(size=3, replace=False)
            jrand = map(np.random.randint, self.ndims)
            xx = x0 + self.factor * (x1 - x2)
            for chromosome, xc, n, r in zip(t, xx, self.ndims, jrand):
                ind = np.random.random(n) < self.cross_prob
                chromosome[ind] = xc[ind]
                chromosome[r] = xc[r]


class DifferentialEvolution1(DifferentialEvolution):
    # As a population of chromosomes

    params = {
        "factor": 0.5,
        "cross_prob": 0.75,
    }

    def init(self):
        self.ndim = len(self[0])

    def move(self):
        self.test = self.copy()
        for t in self.test:
            x0, x1, x2 = choice(self, size=3, replace=False)
            xx = x0 + self.factor * (x1 - x2)
            jrand = np.random.randint(self.ndim)
            ind = np.random.random(n) < self.cross_prob
            t[ind] = xx[ind]
            t[jrand] = xx[jrand]

