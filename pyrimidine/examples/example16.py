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
    def gender(self):
        return self[1][0]


from utils import *
class MyPopulation(SGAPopulation):
    element_class = _Individual

    def mate(self):
        males = [ind for ind in self.individuals if ind.gender==1]
        male_fits = np.array([ind.fitness for ind in self if ind.gender==1])
        ks = np.argsort(male_fits)
        males = [males[k] for k in ks]
        females = [ind for ind in self.individuals if ind.gender==0]
        female_fits = np.array([ind.fitness for ind in self if ind.gender==0])
        ks = np.argsort(female_fits)
        females = [females[k] for k in ks]
        M = len(males)
        F = len(females)

        for k, m in enumerate(males):
            r = randint(0, F-1) / F
            if c(m[1][1:]) <= r:
                f = boltzmann_select(females, female_fits)
                if k / M >=c(f[1][1:]):
                    f.cross(m)

    # def select(self):
    #     self.individuals = self.individuals[:20]

pop = MyPopulation.random(sizes=(8,9), size=20)

stat={'Fitness':'fitness', 'Best Fitness':'best_fitness'}
data = pop.history(stat=stat)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data.index, data['Fitness'], data.index, data['Best Fitness'])
ax.legend(('Fitness', 'Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()


