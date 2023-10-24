#!/usr/bin/env python3

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint
import numpy as np


from pyrimidine.benchmarks.special import *

from digit_converter import *


c=IntervalConverter(-30,30)
n = 8
evaluate = rosenbrock(n=n)

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)

class _Individual(BaseIndividual):
    element_class = _Chromosome
    default_size = n

    def _fitness(self):
        return - evaluate(self.decode())


class SAIndividual(SimulatedAnnealing, _Individual):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        r = randint(0, len(self)-1)
        cpy.chromosomes[r].mutate()
        return cpy


sa = SAIndividual.random(size=n)

sa_data = sa.evolve(stat={'Fitness':'fitness', 'Phantom Fitness':lambda ind: ind.phantom.fitness}, n_iter=200, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
sa_data.loc[40:,['Fitness', 'Phantom Fitness']].plot(ax=ax)
plt.show()

