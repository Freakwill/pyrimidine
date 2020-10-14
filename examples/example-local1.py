#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint


from pyrimidine.benchmarks.special import *

from digit_converter import *


c=IntervalConverter(-30,30)

evaluate = rosenbrock(20)

class _Chromosome(BinaryChromosome):
    def decode(self):
        return c(self)

class _Individual(BaseIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = _Chromosome
    default_size = 20

    def _fitness(self):
        x = [self[k].decode() for k in range(20)]
        return - evaluate(x)



class SAIndividual(_Individual, SimulatedAnnealing):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        r = randint(0, len(self)-1)
        cpy.chromosomes[r].mutate()
        return cpy


class Individual2(MonoIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """
    element_class = FloatChromosome

    def _fitness(self):
        x = self.chromosome
        return - evaluate(x)

class RWIndividual(Individual2, RandomWalk):
    pass


sa = SAIndividual.random(size=20)

sa_data = sa.history()

# rw = RWIndividual.random(size=20)

# rw_data = rw.history(n_iter=500)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
sa_data[['Fitness']].plot(ax=ax)
plt.show()



