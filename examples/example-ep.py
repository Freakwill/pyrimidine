#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *

from pyrimidine.deco import fitness_cache

# generate a knapsack problem randomly

n_bags = 50
_evaluate = Knapsack.random(n_bags=n_bags)

@fitness_cache
class _Individual(BaseEPIndividual):

    element_class = BinaryChromosome // n_bags, FloatChromosome // n_bags

    def decode(self):
        return self.chromosomes[0]

    def _fitness(self):
        return _evaluate(self.decode())

    def mutate(self):
        rx = np.random.rand(*self.chromosomes[0].shape)
        b = np.bool(rx < self.variance)
        self.chromosomes[0][b] = 1-self.chromosomes[0][b]
        
        rv = np.random.randn(*self.variance.shape)
        self.variance += self.c * rv


class _Population(EvolutionProgramming, BasePopulation):

    element_class = _Individual
    default_size = 10


pop = _Population.random()

data = pop.evolve(max_iter=10, period=2, history=True, verbose=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)

data[['Mean Fitness', 'Max Fitness']].plot(ax=ax)
ax.set_xlabel('Generations * 100')
ax.set_ylabel('Fitness')
plt.show()
