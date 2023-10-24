#!/usr/bin/env python3

import numpy as np

from pyrimidine import *
from pyrimidine.benchmarks.optimization import *


# generate a knapsack problem randomly
n_bags = 100
_evaluate = Knapsack.random(n=n_bags)

class _Individual(PolyIndividual[BinaryChromosome.set(default_size=n_bags)]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())


class _Population(HOFPopulation):
    element_class = _Individual
    default_size = 10

class MySpecies(BaseSpecies):
    element_class = _Population
    default_size = 2


sp = MySpecies.random()
stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
data = sp.evolve(stat=stat, n_iter=50, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
