#!/usr/bin/env python3

from itertools import product
import numpy as np

from pyrimidine import *
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle

from digit_converter import unitIntervalConverter


# Generate a knapsack problem randomly
n_bags = 100
_evaluate = Knapsack.random(n=n_bags)

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self.chromosomes[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def expect(self):
        return unitIntervalConverter(self.chromosomes[1])


class _Population(HOFPopulation):
    element_class = _Individual
    default_size = 20


class MySpecies(DualSpecies):
    element_class = _Population

    def match(self, male, female):
        return male.expect <= male.ranking and female.expect <= female.ranking

    @property
    def expect(self):
        return np.mean([ind.expect for ind in self.males] + [ind.expect for ind in self.females])

    @property
    def best_expect(self):
        return self.best_individual.expect


class YourSpecies(DualSpecies):
    element_class = _Population

    def mate(self):
        male_offspring = []
        female_offspring = []
        for male, female in product(self.males, self.females):
            if random()<0.25:
                child = male.cross(female)
                if random()<0.5:
                    male_offspring.append(child)
                else:
                    female_offspring.append(child)

        self.populations[0].individuals += male_offspring
        self.populations[1].individuals += female_offspring


class MyPopulation(_Population):
    default_size = 20


sp = MySpecies.random(sizes=(n_bags, 4))

stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness', 'Mean Expect': 'expect', 'Best Expect': 'best_expect'}

data = sp.evolve(stat=stat, n_iter=100, history=True)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(121)
data[['Male Fitness', 'Female Fitness', 'Best Fitness']].plot(ax=ax)
ax2 = fig.add_subplot(122)
data[['Mean Expect', 'Best Expect']].plot(ax=ax2)
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
