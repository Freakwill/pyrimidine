#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *
from pyrimidine.utils import shuffle

from digit_converter import *

c = unitIntervalConverter

# generate a knapsack problem randomly
_evaluate = Knapsack.random(n=50)

class _Individual(PolyIndividual[BinaryChromosome]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def expect(self):
        return self[1]


class _Population(BasePopulation):
    element_class = _Individual
    default_size = 25

    def select(self):
        ks = np.argsort([individual.fitness for individual in self.individuals])
        self.individuals = [self.individuals[k] for k in ks[-10:]]

class MySpecies(DualSpecies):
    element_class = _Population

    def match(self, male, female):
        mr = self.populations[0].get_rank(male)
        fr = self.populations[1].get_rank(female)
        return c(male.expect) <= fr and c(female.expect) <= mr

    @property
    def expect(self):
        return np.mean([ind.expect for ind in self.males] + [ind.expect for ind in self.females])


sp = MySpecies.random(sizes=(50, 8))

class MyIndividual(MonoBinaryIndividual):
    def _fitness(self):
        return _evaluate(self.chromosome)

    def mate(self, mate_prob=None):
        offspring = [individual.cross(other) for individual, other in zip(self.individuals[::2], self.individuals[1::2])
        if random() < (mate_prob or self.mate_prob)]
        self.individuals += offspring
        X = self.individuals[1::2]
        shuffle(X)
        offspring = [individual.cross(other) for individual, other in zip(self.individuals[::2], X)
        if random() < (mate_prob or self.mate_prob)]
        self.individuals += offspring
    
class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    default_size = 58

pop = MyPopulation(individuals=sp.populations[0].clone().individuals + sp.populations[1].clone().individuals, fitness=sp.fitness)

stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness', 'Mean Expect': 'expect'}

import time

time1 = time.perf_counter()
data = sp.history(stat=stat, n_iter=300)
time2 = time.perf_counter()
print(time2 - time1)

time1 = time.perf_counter()
data2 = pop.history(n_iter=300)
time2 = time.perf_counter()
print(time2 - time1)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data['Best Fitness'].plot(ax=ax)

data2['Best Fitness'].plot(ax=ax, style='--')
ax.legend(('My Fitness', 'Your Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
