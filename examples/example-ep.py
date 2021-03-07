#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly

n_bags = 50
_evaluate = Knapsack.random(n=n_bags)

class _Individual(BaseEPIndividual):
    element_class = BinaryChromosome, FloatChromosome

    def decode(self):
        return self.chromosomes[0]


    def _fitness(self):
        return _evaluate(self.decode())

    def mutate(self):
        rx = np.random.rand(*self.chromosomes[0].shape)
        self.chromosomes[0] ^= (rx < self.variance)
        
        rv = np.random.randn(*self.variance.shape)
        self.variance += self.c * rv


class _Population(EPPopulation, BasePopulation):
    element_class = _Individual
    default_size = 40


pop = _Population.random(sizes=(n_bags, n_bags))

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}

data = pop.evolve(stat=stat, n_iter=100, period=10, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations * 100')
ax.set_ylabel('Fitness')
plt.show()
