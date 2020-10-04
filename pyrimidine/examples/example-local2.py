#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import random


from pyrimidine.benchmarks.special import *

from digit_converter import *


c=IntervalConverter(-10,10)

evaluate = rosenbrock(8)

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)

class _Individual(PolyIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = _Chromosome
    default_size = 8

    def _fitness(self):
        x = [self[k].decode() for k in range(8)]
        return - evaluate(x)


class TSIndividual(_Individual, BaseTabuSearch):
    actions = [k for k in range(8)]

    def move(self, action):
        cpy = self.clone()
        for chromosome in cpy.chromosomes:
            if random() < 0.5:
                chromosome[action] = 1 - chromosome[action]
        return cpy


ts = TSIndividual.random(size=8)

ts_data = ts.history(n_iter=100)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ts_data.index, ts_data['Fitness'])

plt.show()

