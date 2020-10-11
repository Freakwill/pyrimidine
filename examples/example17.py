#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle


# generate a knapsack problem randomly

n_bags = 100
_evaluate = Knapsack.random(n=n_bags)

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())


class _Population(BasePopulation):
    element_class = _Individual
    default_size = 20

class MySpecies(BaseSpecies):
    element_class = _Population


sp = MySpecies.random(sizes=(n_bags, 10))

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
data, t = sp.perf(stat=stat, n_iter=200, n_repeats=1)
print(t)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
