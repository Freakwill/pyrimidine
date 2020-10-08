#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import MonoBinaryIndividual
from pyrimidine.population import *

from pyrimidine.benchmarks.optimization import *

n = 50
_evaluate = Knapsack.random(n)

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        return _evaluate(self.chromosome)

    def dual(self):
        return self.__class__([c.dual() for c in self.chromosomes])
    

# MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))

class _Population1(SGAPopulation):
    element_class = MyIndividual
    default_size = 50

class _Population2(DualPopulation):
    element_class = MyIndividual
    default_size = 50

pop1 = _Population1.random(size=n)
pop2 = _Population2.random(size=n)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness'}
data1 = pop1.history(stat=stat, n_iter=300)
data2 = pop2.history(stat=stat, n_iter=300)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data1[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
data2[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('M1', 'B1', 'M2', 'B2'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
