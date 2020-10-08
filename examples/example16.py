#!/usr/bin/env python3

from pyrimidine import *
import numpy as np
from pyrimidine.benchmarks.optimization import *

from digit_converter import *

c = unitIntervalConverter

# generate a knapsack problem randomly
_evaluate = Knapsack.random(n=50)

class _Individual(MixIndividual[BinaryChromosome, BinaryChromosome]):

    def decode(self):
        return self[0]

    def _fitness(self):
        return _evaluate(self.decode())

    @property
    def expect(self):
        return self[1]


class _Population(BasePopulation):
    element_class = _Individual
    default_size = 20

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
    
class MyPopulation(SGAPopulation):
    element_class = MyIndividual
    default_size = 40

pop = MyPopulation()
pop.individuals = sp.populations[0].individuals + sp.populations[1].individuals


stat={'Male Fitness':'male_fitness', 'Female Fitness':'female_fitness', 'Best Fitness': 'best_fitness', 'Mean Expect': 'expect'}
data = sp.history(stat=stat, n_iter=500)
# data.to_csv('hehe.csv')

data2 = pop.history(n_iter=500)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data['Best Fitness'].plot(ax=ax)

data2['Best Fitness'].plot(ax=ax, style='--')
ax.legend(('Fitness I', 'Fitness II'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
