#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrimidine import MonoBinaryIndividual, AgePopulation, AgeIndividual, SGAPopulation

from pyrimidine.benchmarks.optimization import *

# generate a knapsack problem randomly
evaluate = Knapsack.random(200, W=0.6)

class MyIndividual(AgeIndividual, MonoBinaryIndividual):
    life_span=5
    def _fitness(self):
        return evaluate(self.chromosome)


class MyPopulation(AgePopulation):
    element_class = MyIndividual


class YourPopulation(SGAPopulation):
    element_class = MyIndividual

pop = YourPopulation.random(size=20)

stat={'Mean Fitness':'mean_fitness', 'Best Fitness':'best_fitness'}
data, _ = pop.perf(stat=stat)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)


pop = MyPopulation.random(size=20)

data, _ = pop.perf(stat=stat)

data[['Mean Fitness', 'Best Fitness']].plot(ax=ax)
ax.legend(('Mean Fitness', 'Best Fitness', 'My Fitness', 'My Best Fitness'))
ax.set_xlabel('Generations')
ax.set_ylabel('Fitness')
plt.show()
