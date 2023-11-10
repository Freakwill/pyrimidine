#!/usr/bin/env python

"""
Differential Evolution Algorithm
"""

import copy

import numpy as np
from .mixin import *
from . import MonoIndividual

from .utils import *


class DifferentialEvolution(PopulationModel):

    params ={
    "factor" : 0.5,
    "cross_prob": 0.75,
    }

    test_individuals = []

    def init(self):
        self.dimension = len(self.individuals[0][0])
        self.test = self.clone()

    def transition(self):
        self.move()
        for k, (test_individual, individual) in enumerate(zip(self.test, self)):
            if test_individual.fitness > individual.fitness:
                self.individuals[k] = test_individual

    def move(self):
        for t in self.test:
            x0, x1, x2 = choice(self.individuals, size=3, replace=False)

            jrand = np.random.randint(0, self.dimension)
            xx = x0.chromosome + self.factor * (x1.chromosome - x2.chromosome)
            for j in range(self.dimension):
                if random()<self.cross_prob or j == jrand:
                    t.chromosomes[0][j] = xx[j]
