#!/usr/bin/env python3

from random import random
import numpy as np

from pyrimidine import HybridPopulation, HOFPopulation, BinaryChromosome
from pyrimidine.benchmarks.optimization import *


# generate a knapsack problem randomly
n_bags = 100
_evaluate = Knapsack.random(n_bags)

_Individual = (BinaryChromosome // n_bags).set_fitness(_evaluate)


_Population = HOFPopulation[_Individual] // 5


class _HybridPopulation(HybridPopulation[_Population, _Population, _Individual, _Individual]):

    def max_fitness(self):
        # compute maximum fitness for statistics
        return max(self.get_all_fitness())


sp = _HybridPopulation.random()
data = sp.evolve(max_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of Hybrid Population (Mixed by individuals and populations)')
plt.show()
