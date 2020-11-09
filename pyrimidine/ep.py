#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import BasePopulationModel
from .chromosome import FloatChromosome
from .individual import MixIndividual
from .utils import max_lb

import numpy as np


class BaseEPIndividual(MixIndividual):
    """A particle in EP
    """

    params = {'c':0.1, 'epsilon':0.0001}

    element_class = FloatChromosome, FloatChromosome

    def decode(self):
        return self.chromosomes[0]

    @property
    def variance(self):
        return self.chromosomes[1]

    @variance.setter
    def variance(self, v):
        self.chromosomes[1] = v
    

    def mutate(self):
        rx = np.random.rand(*self.chromosomes[0].shape)
        rv = np.random.rand(*self.variance.shape)
        self.chromosomes[0] += rx * np.sqrt(self.variance)
        self.variance += self.c * rv * np.sqrt(self.variance)
        self.variance = max_lb(self.epsilon)(self.variance)


class EPPopulation(BasePopulationModel):
    element_class = BaseEPIndividual

    # def select(self):
    #     self.individuals = self.get_best_individuals(0.5)
    #     print(len(self.individuals))

    def transit(self, *args, **kwargs):
        cpy = self.clone()
        self.mutate()
        self.merge(cpy)
        self.select()

