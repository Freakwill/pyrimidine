#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle

from digit_converter import *

c = unitIntervalConverter

# generate a knapsack problem randomly

n_bags = 100
_evaluate = Knapsack.random(n=n_bags)

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())


class _Population(BasePopulation):
    element_class = _Individual
    default_size = 20


class MySpecies(DualSpecies):
    element_class = _Population

    def mate(self):
        male_offspring = []
        female_offspring = []
        children = [male.cross(female) for male, female in zip(self.males, self.females) if random()<0.75]
        male_offspring.extend(children[::2])
        female_offspring.extend(children[1::2])
        for _ in range(1):
            shuffle(self.females)
            children = [male.cross(female) for male, female in zip(self.males, self.females) if random()<0.75]
            male_offspring.extend(children[::2])
            female_offspring.extend(children[1::2])

        self.populations[0].individuals += male_offspring
        self.populations[1].individuals += female_offspring

class MyPopulation(_Population):
    default_size = 40

    def mate(self):
        # super(MyPopulation, self).mate()
        shuffle(self.individuals)
        super(MyPopulation, self).mate()



sp = MySpecies.random(size=n_bags)
pop = MyPopulation(individuals=sp.populations[0].clone().individuals+sp.populations[1].clone().individuals)

stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness'}
data2, t2 = sp.perf(stat=stat, n_iter=200, n_repeats=10)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness': 'best_fitness'}
data3, t3 = pop.perf(stat=stat, n_iter=200, n_repeats=10)


print(t2, t3)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data2['Best Fitness'].plot(ax=ax, style='--')
data3['Best Fitness'].plot(ax=ax, style='-.')
ax.legend(('My Fitness', 'Traditional Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
