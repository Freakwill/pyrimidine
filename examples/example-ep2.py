#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.special import *


# generate a knapsack problem randomly

n=10
_evaluate = schaffer

class _Individual(BaseEPIndividual):
    element_class = FloatChromosome, FloatChromosome

    def decode(self):
        return self.chromosomes[0]


    def _fitness(self):
        return - _evaluate(self.decode())


class _Population(EPPopulation, BasePopulation):
    element_class = _Individual
    default_size = 20


pop = _Population.random(sizes=(n, n))

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
data = pop.evolve(stat=stat, n_iter=500, period=5, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations * 5')
ax.set_ylabel('Fitness')
plt.show()
