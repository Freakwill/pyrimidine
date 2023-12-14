#!/usr/bin/env python3

from random import random
import numpy as np

from pyrimidine import MultiPopulation, HOFPopulation, MonoIndividual, BinaryChromosome
from pyrimidine.benchmarks.optimization import *


# generate a knapsack problem randomly
n_bags = 100
_evaluate = Knapsack.random(n_bags)

class _Individual(MonoIndividual[BinaryChromosome // n_bags]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def max_fitness(self):
        return self.fitness

    @property
    def mean_fitness(self):
        return self.fitness


class _Population(HOFPopulation):
    element_class = _Individual
    default_size = 10


class HybridPopulation(MultiPopulation):
    element_class = (_Population //2, _Individual//2)

    def migrate(self, migrate_prob=None, copy=True):
        migrate_prob = migrate_prob or self.migrate_prob
        for this, other in zip(self[:-1], self[1:]):
            if random() < migrate_prob:
                if isinstance(this, _Population):
                    this_best = this.get_best_individual(copy=copy)
                    if isinstance(other, _Population):
                        other.append(this.get_best_individual(copy=copy))
                        this.append(other.get_best_individual(copy=copy))
                    else:
                        this.append(other.copy())
                else:
                    this_best = this.copy()
                    if isinstance(other, _Population):
                        other.append(this.get_best_individual(copy=copy))

    def transition(self, *args, **kwargs):
        for p in self:
            p.transition(*args, **kwargs)
        self.migrate()

    # def max_fitness(self):
    #     return max(e.fitness for e in self)


sp = HybridPopulation.random()
data = sp.evolve(n_iter=100, history=True)


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
ax.set_title('Demo of Hybrid Population (Mixed by individuals and populations)')
plt.show()
