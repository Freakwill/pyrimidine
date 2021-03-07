#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base import *
from .errors import *
from .gene import *
from .chromosome import *
from .individual import *
from .population import *
from .species import *
from .saga import *
from .pso import *
from .ep import *


__template__ = """
from pyrimidine import MonoBinaryIndividual
from pyrimidine.population import SGA2Population

from pyrimidine.benchmarks.optimization import *

n = 50
_evaluate = Knapsack.random(n)


# Define individual
class MyIndividual(MonoBinaryIndividual):
    def _fitness(self) -> float:
        # To evaluate an individual!
        return _evaluate(self.chromosome)
    
# MyIndividual = MonoBinaryIndividual.set_fitness(lambda o: _evaluate(o.chromosome))

# Define Population
class MyPopulation(SGA2Population):
    element_class = MyIndividual
    default_size = 20

# MyPopulation = SGA2Population[MyIndividual] // 20

pop = MyPopulation.random(size=n)

stat={'Mean Fitness':'fitness', 'Best Fitness':'best_fitness', 'Standard Deviation of Fitnesses': 'std_fitness'}
data = pop.evolve(stat=stat, n_iter=200, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(loc='upper left')
data['Standard Deviation of Fitnesses'].plot(ax=ax2, style='r--')
ax2.legend(loc='lower right')
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
"""