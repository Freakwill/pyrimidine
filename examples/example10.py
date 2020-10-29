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
        return MyIndividual([c.dual() for c in self])
    

# MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))

pop = MyIndividual.random(size=n) * 20


stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness'}
data = pop.evolve(stat=stat, n_iter=200, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
