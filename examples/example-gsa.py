#!/usr/bin/env python3

from pyrimidine.base import BasePopulation
from pyrimidine.gsa import Particle, GravitySearch

from pyrimidine.benchmarks.special import *

# generate a knapsack problem randomly
def evaluate(x):
    return -rosenbrock(8)(x)

class _Particle(Particle):
    default_size = 8
    def _fitness(self):
        return evaluate(self.position)


class MyGravitySearch(GravitySearch, BasePopulation):
    element_class = _Particle
    default_size = 40

pop = MyGravitySearch.random()


stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=200, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data.loc[10:, ['Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
