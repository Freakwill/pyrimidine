#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *

from digit_converter import *

c = unitIntervalConverter

# generate a knapsack problem randomly
evaluate = Knapsack.random()

class _Individual(MixIndividual[BinaryChromosome, BinaryChromosome]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return evaluate(self.decode())

    @property
    def expect(self):
        return self[1]


class _Population(BasePopulation):
    element_class = _Individual
    default_size = 20

class MySpecies(DualSpecies):
    element_class = _Population

    def match(self, male, female):
        mr = self.populations[0].get_rank(male)
        fr = self.populations[1].get_rank(female)
        return c(male.expect) <= fr and c(female.expect) <= mr


sp = MySpecies.random(sizes=(20, 8))


stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness'}
data = sp.history(stat=stat)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.index, data['Male Fitness'], data.index, data['Female Fitness'], data.index, data['Best Fitness'])
ax.legend(('Male Fitness', 'Female Fitness', 'Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()


