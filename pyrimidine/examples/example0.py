#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import MonoBinaryIndividual, SGAPopulation, BinaryChromosome

from pyrimidine.benchmarks.optimization import *

_evaluate = Knapsack.random(n=20)

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        return _evaluate(self.chromosome)


class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    default_size = 20

pop = MyPopulation.random(size=20)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness'}
data = pop.history(stat=stat, n_iter=100)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
