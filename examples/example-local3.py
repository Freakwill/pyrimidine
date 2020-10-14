#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import *
from pyrimidine.local_search import *
from random import randint


from pyrimidine.benchmarks.optimization import *

from digit_converter import *


evaluate = Knapsack.random()


class _Individual(MonoBinaryIndividual):
    """base class of individual

    You should implement the methods, cross, mute
    """

    def _fitness(self):
        return evaluate(self.chromosome)


class MyIndividual(SimulatedAnnealing, _Individual):

    def get_neighbour(self):
        cpy = self.clone(fitness=None)
        cpy.chromosome.mutate()
        return cpy


i = MyIndividual.random(size=20)

data = i.get_history(n_iter=300)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data.plot(ax=ax)
plt.show()

