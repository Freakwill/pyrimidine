#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BasePopulationModel, BaseChromosome
from .chromosome import FloatChromosome
from .individual import MixIndividual
from .utils import max_lb
from random import choice
from toolz.itertoolz import groupby
from operator import attrgetter

import numpy as np


class BaseEPIndividual(MixIndividual):
    """A particle in EP
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
    

    def mutate(self):
        rx = np.random.randn(*self.chromosomes[0].shape)
        self.chromosomes[0] += rx * np.sqrt(self.variance)

        rv = np.random.randn(*self.variance.shape)
        self.variance += self.c * rv * np.sqrt(self.variance)
        self.variance = max_lb(self.epsilon)(self.variance)


class EPPopulation(BasePopulationModel):
    element_class = BaseEPIndividual

    def select(self):
        d = groupby(attrgetter('fitness'), self.sorted_individuals)
        inds = []
        ks = np.sort(list(d.keys()))[::-1]
        while len(inds) < self.n_individuals:
            for k in ks:
                if d[k]:
                    a = choice(d[k])
                    inds.append(a)
                    d[k].remove(a)
        self.individuals = inds


    def transit(self, *args, **kwargs):
        cpy = self.clone()
        self.mutate()
        self.merge(cpy)
        self.select()
